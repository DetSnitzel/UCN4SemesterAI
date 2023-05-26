import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import os
import keras.backend as K
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Activation,
    Conv2D,
    MaxPooling2D,
    Reshape,
    Lambda,
    LSTM,
    Bidirectional,
    TimeDistributed,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization

## Datareading
y_train = pd.read_csv("./namedataset/written_name_train_v2.csv")
y_test = pd.read_csv("./namedataset/written_name_test_v2.csv")
y_validation = pd.read_csv("./namedataset/written_name_validation_v2.csv")

##Show some of this dataset
plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = './namedataset/train_v2/train/'+y_train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    plt.title(y_train.loc[i, 'IDENTITY'], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)
plt.show()


##Prints how many null values we have in identity in each dataset
print("Number of nulls in training set   : ", y_train["IDENTITY"].isnull().sum())
print("Number of nulls in test set       : ", y_test["IDENTITY"].isnull().sum())
print("Number of nulls in validation set : ", y_validation["IDENTITY"].isnull().sum())

##Drops all the null value, to clean up the dataset
y_train.dropna(axis=0, inplace=True)
y_test.dropna(axis=0, inplace=True)
y_validation.dropna(axis=0, inplace=True)

##Checking for unreadable images and putting them in an array for themself to show later
unreadable = y_train[y_train["IDENTITY"] == "UNREADABLE"]
unreadable.reset_index(inplace=True, drop=True)

##Shows the unreadable training images
plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = './namedataset/train_v2/train/'+unreadable.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    plt.title(unreadable.loc[i, 'IDENTITY'], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)
plt.show()

##Overrides out old array to only have labels for the readable images
y_train = y_train[y_train["IDENTITY"] != "UNREADABLE"]
y_test = y_test[y_test["IDENTITY"] != "UNREADABLE"]
y_validation = y_validation[y_validation["IDENTITY"] != "UNREADABLE"]

##Uppercases the Identity column
y_train["IDENTITY"] = y_train["IDENTITY"].str.upper()
y_test["IDENTITY"] = y_test["IDENTITY"].str.upper()
y_validation["IDENTITY"] = y_validation["IDENTITY"].str.upper()

##Resets the index of the array to not have the indexes where we have removed labels
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
y_validation.reset_index(inplace=True, drop=True)


#Images reprocessing

def preprocess(img):
    """A fucntion which takes an image and preprocess the image into a 64 pixel height and 256 pixel width and pads the rest to fill.

    Args:
        img (_type_): The image you want preproccessed

    Returns:
        _type_: The preproccessed image in a numpy array, for easy storing of the image and small size when handeling vast amounts of images.
    """
    target_height = 64
    target_width = 256
    
    img = np.expand_dims(img, axis=-1)
    
    final_img = tf.image.resize(
        img, size=(target_height, target_width), preserve_aspect_ratio=True)
    
    final_img = tf.image.pad_to_bounding_box(
        final_img, 0, 0, target_height, target_width)

    return final_img.numpy()


##Size of the dataset we train, test and validate on.
training_size = 100_000
test_size = 10000
validation_size = 3000

##Creates a new array and if there already is a file with preproccesed training images it will load that into the array
##otherwise it will preprocess the training images and then put it into the array and save it as a numpy file for later use.
x_train = []
if os.path.exists("./namedataset/train_preprocessed_images.npy"):
    x_train = np.load("./namedataset/train_preprocessed_images.npy")
    print("X_train length: " + str(len(x_train)))
    print("Training size: " + str(training_size))
else:
    print("Preprocessing images...")

    for i in range(training_size):
        img_dir = "./namedataset/train_v2/train/" + y_train.loc[i, "FILENAME"]
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        image = preprocess(image)
        x_train.append(image.numpy())

    np.save("./namedataset/train_preprocessed_images.npy", x_train)

    print("Done preprocessing training images")
##If you have preproccesed more images than the size of your training_size is set to this will cut it out to avoid any ambiguity later on.      
x_train = x_train[0:training_size]

##Creates a new array and if there already is a file with preproccesed test images it will load that into the array
##otherwise it will preprocess the test images and then put it into the array and save it as a numpy file for later use.
x_test = []
if os.path.exists("./namedataset/test_preprocessed_images.npy"):
    x_test = np.load("./namedataset/test_preprocessed_images.npy")
    print("Test length: " + str(len(x_test)))
    print("Test size: " + str(test_size))
else:
    print("Preprocessing test images...")

    for i in range(test_size):
        img_dir = "./namedataset/test_v2/test/" + y_test.loc[i, "FILENAME"]
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        image = preprocess(image)
        x_test.append(image.numpy())

    np.save("./namedataset/test_preprocessed_images.npy", x_test)

    print("Done preprocessing testing images")
    
##If you have preproccesed more images than the size of your test_size is set to this will cut it out to avoid any ambiguity later on.    
x_test = x_test[0:test_size]

x_validation = []
if os.path.exists("./namedataset/validation_preprocessed_images.npy"):
    x_validation = np.load("./namedataset/validation_preprocessed_images.npy")
    print("X_validation length: " + str(len(x_validation)))
    print("Validation size: " + str(validation_size))
    # print(x_train[0])
else:
    print("Preprocessing validation images...")

    for i in range(validation_size):
        img_dir = (
            "./namedataset/validation_v2/validation/" + y_validation.loc[i, "FILENAME"]
        )
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        image = preprocess(image)
        x_validation.append(image.numpy())

    np.save("./namedataset/validation_preprocessed_images.npy", x_validation)

    print("Done preprocessing validation images")
    
x_validation = x_validation[0:validation_size]

x_train = np.array(x_train).reshape(-1, 64, 256,  1)
x_test = np.array(x_test).reshape(-1, 64, 256, 1)
x_validation = np.array(x_validation).reshape(-1, 64, 256, 1)

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 34  # max length of input labels
num_of_characters = len(alphabets) + 1  # +1 for ctc pseudo blank. A pseudo blank is used in CTC for encoding duplicate characters. And will be removed later when decoding.
num_of_timestamps = 64  # max length of predicted labels

##Displays the training images after being preprocess to make sure it looks alright
plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train.loc[i, "IDENTITY"], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)
plt.show()

def label_to_num(label):
    """Takes a label and converts it to a numpy array corresponding to the label

    Args:
        label (_type_): The label you want converted to a numpy array

    Returns:
        _type_: A numpy array of the label
    """
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))

    return np.array(label_num)


def num_to_label(num):
    """Takes a numpy array and converts it to a label. This is used when the AI has predicted an image. It gives a numpy array of number that we will have to
    convert into a label

    Args:
        num (_type_): The predicted numpy array the AI returns

    Returns:
        _type_: A label
    """
    ret = ""
    for ch in num:
        if ch == -1:  # -1 for the CTC Blank we added earlier
            break
        else:
            ret += alphabets[ch]
    return ret


## TRAINING SET
##Creates what the CTC needs. This is dummy values and are matrixes with ones and zeros
train_y = np.ones([training_size, max_str_len]) * -1
train_label_len = np.zeros([training_size, 1])
train_input_len = np.ones([training_size, 1]) * (num_of_timestamps - 2)
train_output = np.zeros([training_size])
##Attaches the labels to the right image
for i in range(training_size):
    train_label_len[i] = len(y_train.loc[i, "IDENTITY"])
    train_y[i, 0 : len(y_train.loc[i, "IDENTITY"])] = label_to_num(
        y_train.loc[i, "IDENTITY"]
    )

## TESTING SET
##Creates what the CTC needs. This is dummy values and are matrixes with ones and zeros
test_y = np.ones([test_size, max_str_len]) * -1
test_label_len = np.zeros([test_size, 1])
test_input_len = np.ones([test_size, 1]) * (num_of_timestamps - 2)
test_output = np.zeros([test_size])
##Attaches the labels to the right image
for i in range(test_size):
    test_label_len[i] = len(y_test.loc[i, "IDENTITY"])
    test_y[i, 0 : len(y_test.loc[i, "IDENTITY"])] = label_to_num(
        y_test.loc[i, "IDENTITY"]
    )
    
## VALIDATION SET
##Creates what the CTC needs. This is dummy values and are matrixes with ones and zeros
validation_y = np.ones([validation_size, max_str_len]) * -1
validation_label_len = np.zeros([validation_size, 1])
validation_input_len = np.ones([validation_size, 1]) * (num_of_timestamps - 2)
validation_output = np.zeros([validation_size])
##Attaches the labels to the right image
for i in range(validation_size):
    validation_label_len[i] = len(y_validation.loc[i, "IDENTITY"])
    validation_y[i, 0 : len(y_validation.loc[i, "IDENTITY"])] = label_to_num(
        y_validation.loc[i, "IDENTITY"]
    )

##Defines the input shape for the model
input_shape = (64, 256, 1)

##Define the input layer of the model
inputs = Input(shape=input_shape, name="input")

##Convolutional layers which the image will be sent through before going into the lstm layers. This is the feature extraction part of the AI.
conv1 = Conv2D(128, (3,3), padding="same", kernel_initializer='he_normal')(inputs)
batch1 = BatchNormalization()(conv1)
activation1 = Activation('relu')(batch1)
pool1 = MaxPooling2D(pool_size=(2, 2))(activation1)

conv2 = Conv2D(64, (3,3), padding="same", kernel_initializer='he_normal')(pool1)
batch2 = BatchNormalization()(conv2)
activation2 = Activation('relu')(batch2)
pool2 = MaxPooling2D(pool_size=(2, 2))(activation2)
dropout1 = Dropout(0.3)(pool2)

conv3 = Conv2D(256, (3,3), padding="same", kernel_initializer='he_normal')(dropout1)
batch3 = BatchNormalization()(conv3)
activation3 = Activation('relu')(batch3)
pool3 = MaxPooling2D(pool_size=(2, 2))(activation3)
dropout2 = Dropout(0.3)(pool3)

##Reshape the output from the convolutional layers for input into the recurrent layers
reshaped = Reshape(target_shape=((64, 1024)))(dropout2)
dense1 = Dense(128, activation = 'relu', kernel_initializer='he_normal')(reshaped)
##A series of LSTM layers with 256 neurouns. This part learn the language and corrects misspelled words as good as it has been taught. (Good LSTM layers requires good data)
lstm1 = Bidirectional(LSTM(256, return_sequences=True))(dense1)
lstm2 = Bidirectional(LSTM(256, return_sequences=True))(lstm1)
lstm3 = Bidirectional(LSTM(256, return_sequences=True))(lstm2)

##Apply a dense layer to convert the output of the recurrent layers into a sequence of probabilities
dense2 = Dense(num_of_characters)(lstm3)
##The output layer which will give us the prediction
y_pred = Activation('softmax', name='softmax')(dense2)

##How many images will be taken at a time when training
batch_size = 128 

##The shape of labels
labels_shape = [max_str_len]

##This part takes care of the possibility of duplicate characters and will ajust to make the right amount of characters
def ctc_lambda_func(args):
    y_pred, labels, input_lengths, label_lengths = args
    ##The 2 is critical here since the first couple outputs of the RNN
    ##tend to be bad
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_lengths, label_lengths)


##Define the model inputs
labels = Input(name="the_labels", shape=labels_shape, dtype="float32",)
input_lengths = Input(name="input_lengths", shape=[1], dtype="int64")
label_lengths = Input(name="label_lengths", shape=[1], dtype="int64")

##The ctc function from before is being used with a lambda here for later use of a variable instead of a variable instead of a function.
ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_lengths, label_lengths])

##Defines the model inputs and outputs
model = Model(inputs=[inputs, labels, input_lengths, label_lengths], outputs=ctc_loss)

##The model is compiled and given an optimizer and a metrics which is used to save the accuracy of the model when training, which we will be able to use later
model.compile(loss={"ctc":lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

##Print the model summary
model.summary()

##This is the function that trains the whole AI. Its a variable called history which will be used alongside the accuracy metrics from before to create graphs.
##We give the function our images and labels. Then the batch size from before to determine of many image it trains on at a time.
##Epochs is how many iterations it will go through all the data and train on it and learn
##The validatation data given is our validation images and validation labels which it will test and at the end of each iterations/epoch.
##This is done so that you are able to see if the model is beginning to overfit or underfit. We do this give it some data it hasn't trained on otherwise the information
##would be missleading.
history = model.fit(
            x=[x_train, train_y, train_input_len, train_label_len],
            y=np.zeros(len(x_train)),
            batch_size=batch_size,
            epochs=60,
            validation_data=([x_validation, validation_y, validation_input_len, validation_label_len], np.zeros(len(x_validation)))
)

##After the model is trained we save it for later use.
model.save("./crnn.h5", overwrite=True)

##Displays a graph of accuracy over epochs. Both our training accuracy and validation accuracy will be displayed here.
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")

##Displays the loss over epochs. This is closely related to the graph before.
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")

##Load a model
loaded_model = tf.keras.models.load_model('./crnn.h5', compile=False)

##Prints the summary of the loaded model
loaded_model.summary()

##Gets the input and output model of the loaded model. (This is necessary to have otherwise you won't be able to predict with the loaded model)
loaded_model = tf.keras.models.Model(loaded_model.get_layer(name="input").input, loaded_model.get_layer(name="softmax").output)

##Uses the loaded model to predict the validation set.
preds = loaded_model.predict(x_validation)
##Decodes the arrays of number given by the prediction
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)[0][0])

##Creates an array for the predictions
prediction = []
##The array of decoded predictions will be made into readable labels and added to the array of predictions.
for i in range(validation_size):
    prediction.append(num_to_label(decoded[i]))

##The true label of the image to be able to compare the results
y_true = y_validation.loc[0:validation_size, 'IDENTITY']
##Initializing of variables
correct_char = 0
total_char = 0
correct = 0

##Calculates the correct charaters and words predicted. The correct characters will have to be taken with a grain of salt because if the AI predicts a word like Anna
##as AAAA it will have 50% of the characters right but the words is way to wrong to give anything.
for i in range(validation_size):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
            
    if pr == tr :
        correct += 1 
##Print the calculated values of the predictions    
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/validation_size))

##Reads the csv for labels. (This is done before, but this is able to be done in a seperate file which is why it is in here)
test = pd.read_csv('./namedataset/written_name_test_v2.csv')

##Predicts and displays the 36 first predicted images and the predicted labels as label above each image
plt.figure(figsize=(15, 10))
for i in range(36):
    ax = plt.subplot(6, 6, i+1)
    img_dir = './namedataset/test_v2/test/'+test.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    print(image.shape)
    image = preprocess(image)
    print(image.shape)
    
    pred = loaded_model.predict(image.reshape(1, 64, 256, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])
    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
    
plt.subplots_adjust(wspace=0.2, hspace=-0.8)