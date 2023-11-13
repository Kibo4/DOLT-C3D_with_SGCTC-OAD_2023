#%%
import matplotlib.pyplot as plt
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

#%%
### import data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
    path='mnist.npz',
)
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
#%% md
# createan artifical sequence ofdata
#%%
nbElemPerSeq = 20
nbForInput = 80
#%%

def createArtficial(set, size):
    set_seq = tf.cast(tf.reshape(set, shape=[size // nbElemPerSeq, nbElemPerSeq, 28, 28, 1]), tf.float32)
    set_seq = tf.transpose(set_seq, [0, 2, 1, 3, 4])  #[size//nbElemPerSeq,28,nbElemPerSeq,28,1]
    set_seq = tf.reshape(set_seq, [size // nbElemPerSeq, 28, nbElemPerSeq * 28, 1])
    set_seq = tf.pad(set_seq, [[0, 0], [0, 0], [0, 0], [0, 0]])
    # print(tf.shape(set_seq))
    set_seq = tf.reshape(set_seq, [size // nbElemPerSeq, 28, nbForInput, tf.shape(set_seq)[2] // nbForInput, 1])
    set_seq = tf.transpose(set_seq, [0, 2, 1, 3, 4])
    # print(tf.shape(set_seq))
    return set_seq


x_train_seq = createArtficial(x_train, 60000)
x_test_seq = createArtficial(x_test, 10000)


y_train_seq = tf.reshape(y_train, shape=[60000 // nbElemPerSeq, nbElemPerSeq])
y_test_seq = tf.reshape(y_test, shape=[10000 // nbElemPerSeq, nbElemPerSeq])
#cast test to int32
y_train_seq = tf.cast(y_train_seq, tf.int32)+1
y_test_seq = tf.cast(y_test_seq, tf.int32)+1



nbOfFramesPerLabel = (nbForInput)/nbElemPerSeq
# first label is between 0 and nbOfFramesPerLabel, second between nbOfFramesPerLabel and 2*nbOfFramesPerLabel, etc
ranging = tf.range(nbElemPerSeq, dtype=tf.float32)
# get the tensor of the start frame of each label
startFrame = ranging * (nbOfFramesPerLabel) # shape is [nbForInput]
# get the tensor of the end frame of each label
endFrame = (ranging + 1) * (nbOfFramesPerLabel) # shape is [nbForInput]


concat = tf.concat([startFrame[ :,tf.newaxis], endFrame[:,tf.newaxis]], axis=-1)

#round the start and end frames
concat = tf.cast(tf.round(concat), tf.int32)
# repeat for all seqeunces
concattrain = tf.repeat(concat[tf.newaxis], repeats=60000 // nbElemPerSeq, axis=0)
concattest = tf.repeat(concat[tf.newaxis], repeats=10000 // nbElemPerSeq, axis=0)

y_train_seq = y_train_seq[:, :, tf.newaxis]
y_test_seq = y_test_seq[:, :, tf.newaxis]

#concat to the y
y_train_seq = tf.concat([y_train_seq, concattrain], axis=-1)
y_test_seq = tf.concat([y_test_seq, concattest], axis=-1)
# y_train_seq shape is [60000//nbElemPerSeq,nbElemPerSeq,3]
# y_test_seq shape is [10000//nbElemPerSeq,nbElemPerSeq,3]


batch = 4
train_ds = tf.data.Dataset.from_tensor_slices((x_train_seq, y_train_seq))
train_ds = train_ds.shuffle(10000)
train_ds = train_ds.padded_batch(batch)
train_ds = train_ds.repeat()

test_ds = tf.data.Dataset.from_tensor_slices((x_test_seq, y_test_seq)).batch(batch)
a = next(iter(train_ds))
print(tf.shape(a[0]))
print(tf.shape(a[1]))
#%%
sample = 7
_, axs = plt.subplots(2, nbForInput // 2, figsize=(28, 5))
axs = axs.flatten()
currentframe=0
currentId=0
for img, ax in zip(x_test_seq[sample, :], axs):
    ax.imshow(img)
    # for the correponding label using start and end frames
    if currentframe>=y_test_seq[sample,currentId,2]:
        currentId+=1
    currentframe+=1
    ax.set_title(y_test_seq[sample,currentId,0].numpy()-1)
    #remove x and yticks
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

#%%
### Define the evaluate/stats
def EvalAndStat(model, test_seqs, labels_test_seqs):
    PurcentOfBlankPerSeq = 0
    numberofSequence = len(test_seqs)

    correctPred = 0
    correctLen = 0

    for id, sample in enumerate(test_seqs):
        results = model(sample[tf.newaxis]) # [1,nbForInput,nbClass+1]
        # results = tf.squeeze(results, axis=0)
        resultsTransposed = tf.transpose(results, [1, 0, 2])
        resultBrut = tf.argmax(resultsTransposed, axis=-1).numpy()
        PurcentOfBlankPerSeq += len(tf.where(resultBrut == 0)) / len(resultBrut)
        # print(PurcentOfBlankPerSeq)
        # print(resultBrut)
        label = labels_test_seqs[id][:, 0]
        res = tf.nn.ctc_greedy_decoder(resultsTransposed, [nbForInput] * 1, blank_index=0)[0][0].values.numpy()
        # print(res)
        # print(label)
        # print(tf.where(res==label))
        if len(res) != len(label):
            if len(res) < len(label):
                res = tf.pad(res, [[len(label) - len(res), 0]])
            else:
                label = tf.pad(label, [[len(res) - len(label), 0]])
            # print("corrected")
            # print(len(res))
            # print(len(label))
        correctPred += len(tf.where(tf.cast(res, tf.int64) == tf.cast(label, tf.int64)))
        correctLen += len(res)
        # break
    print("Purcent of blank per seq : ", PurcentOfBlankPerSeq / numberofSequence * 100)
    print("ExactPredictions : ", correctPred / correctLen)


nbClass=10
##Create simple model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # inputs = tf.keras.Input(shape=(nbElemPerSeq,28, 28,1))
        self.convs = [tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(nbElemPerSeq, 28, 12, 1),
                                    padding="same")] +\
                     [tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding="same") for _ in range(4)]

        self.maxpools = [tf.keras.layers.MaxPooling3D((1, 2, 2), padding="same") for _ in range(5)]
        self.flat = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(nbClass + 1, activation="linear")  #batch, x,y, features
        # self.reluShifted = tf.keras.layers.ReLU(threshold=1e-5)
    # self.logSoftmax = layers.Softmax()


    def call(self, x):
        for id, conv in enumerate(self.convs):
            x = conv(x)
            x = self.maxpools[id](x)
        x = self.flat(x)  #32,50,-1
        x = self.dense1(x)  #32,50,128
        x = self.dense2(x)  #32,50,128
        x = self.dense3(x)  #32,50,nbClass+1
        return x


model = MyModel()
out = model(a[0])
print(tf.shape(out))

#%% md
# Define the losses
#%%
from Tools.LossesAndMetrics.ctc_normal_customRec_tf import ctc_loss_log_custom_prior_with_recMatrix_computation # our SG guided ctc + weighted label prior (psi)


def lossSimpleCTC(true,pred):
    shapeOfTrue = tf.shape(true) #[batch,nbElemPerSeq,3]
    shapeOfPred = tf.shape(pred) #[batch,nbElemForInput,11]
    return tf.nn.ctc_loss(true[:,:,0],pred,
                          label_length=tf.repeat(shapeOfTrue[1],repeats = shapeOfTrue[0]),
                          logit_length=tf.repeat(shapeOfPred[1],repeats =shapeOfTrue[0]),
                        logits_time_major=False,blank_index=0)


def lossSegmentationGuidedAndLabelPrior(true,pred,psi=0):
    shapeOfTrue = tf.shape(true) #[batch,nbElemPerSeq]
    shapeOfPred = tf.shape(pred) #[batch,nbElemForInput,11]  # here the length of the sequence is always nbForInput, just repeat itf or each batch, but the loss support variable length

    nbForInputPerBatch = tf.repeat(shapeOfPred[1],repeats = tf.shape(true)[0],axis=0)[:,tf.newaxis]
    # the number of token per sequence is always the same here : nbElemPerSeq, but the loss support variable length
    nbElemPerSeqPerBatch = tf.repeat(shapeOfTrue[1],repeats = tf.shape(true)[0],axis=0)[:,tf.newaxis]

    return ctc_loss_log_custom_prior_with_recMatrix_computation(pred,true,nbForInputPerBatch,nbElemPerSeqPerBatch,weightPrior=psi,doSSG=True)
#%% md
# Define the optimizer and train the model
#%% md
## Version normalCTC
#%%
adam = tf.keras.optimizers.Adam(learning_rate=0.001,    name='Adam')
# lengthLabels = tf.convert_to_tensor([nbElemPerSeq]*batch)[:,tf.newaxis]
# nbForInputLength = tf.convert_to_tensor([nbForInput]*batch)[:,tf.newaxis]
modelNormal = MyModel()

modelNormal.compile(optimizer=adam,
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              # loss=[lambda true,pred:lossSegmentationGuidedAndLabelPrior(true,pred,psi=0)],
              loss=[lossSimpleCTC],
              # loss=lambda true,pred:ctc_ent_loss_log(pred,true+1,nbElemPerSeq),
              metrics=[])
checkpoint_filepath = ".data/models/simpleCTC"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

#%%

historyNormalCTC = modelNormal.fit(train_ds, epochs=10,callbacks=[model_checkpoint_callback],steps_per_epoch=60000/nbForInput/batch,
                    validation_data=test_ds,verbose=2)
#%%
EvalAndStat(model,x_test_seq,y_test_seq)
#%%
# plot the result on one sequence
sample = 7
_, axs = plt.subplots(2, nbForInput // 2, figsize=(28, 5))
axs = axs.flatten()
currentframe=0
currentId=0

#infer the sample
out = model(x_test_seq[sample, tf.newaxis])[0]
out = tf.argmax(out, axis=-1)

for img, ax in zip(x_test_seq[sample, :], axs):
    ax.imshow(img)
    # for the correponding label using start and end frames
    ax.set_title(out[currentframe].numpy()-1)
    currentframe+=1
    #remove x and yticks
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
#%% md
# With Guided and label prior
#%%dam = tf.keras.optimizers.Adam(learning_rate=0.001,    name='Adam')
# lengthLabels = tf.convert_to_tensor([nbElemPerSeq]*batch)[:,tf.newaxis]
# nbForInputLength = tf.convert_to_tensor([nbForInput]*batch)[:,tf.newaxis]
modelsGuided = [MyModel() for i in range(5)]
for id, mod in enumerate(modelsGuided):
    mod.compile(optimizer=adam,
                  loss=[lambda true,pred:lossSegmentationGuidedAndLabelPrior(true,pred,psi=id*0.2)],
                  # loss=lambda true,pred:ctc_ent_loss_log(pred,true+1,nbElemPerSeq),
                  metrics=[])
    checkpoint_filepath = f".data/models/guidedCTC{id*0.2}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)


#%%

#%%
histories = []

for id, mod in enumerate(modelsGuided):
    print("Training model with id ",id,"psi = ",id*0.2)
    histories.append(mod.fit(train_ds, epochs=10,callbacks=[model_checkpoint_callback],steps_per_epoch=60000/nbForInput/batch,
                        validation_data=test_ds,verbose=2))



for id, mod in enumerate(modelsGuided):
    print("Eval model with id ",id,"psi = ",id*0.2)
    EvalAndStat(mod,x_test_seq,y_test_seq)
