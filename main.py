from telegram import *
from telegram.ext import *
import emoji
import cv2
from io import BytesIO
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import json


BOT_TOKEN = "6109508255:AAEsKGYVyq27lFF7XgCm6vqdeBOaBllxlns"
BOT_USERNAME = '@birds_species_bot'
global flag
flag = 3
# Commands


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(emoji.emojize("Hello I am BirdBot :grinning_face: \n Please give me Image to find "
                                                  "the Bird / Animal Species :eagle: :peacock:\n\n Please type "
                                                  ":right_arrow: /start :left_arrow: to start the BOT :eagle:\n\n" 
                                                  " Please type :right_arrow: /image_data :left_arrow: to start the "
                                                  "getting species information \n Please type :right_arrow: animal "
                                                  ":left_arrow: for classification of Animal species"
                                                  "\n\n Please type :right_arrow: bird :left_arrow: for classification "
                                                  "of Bird Species"))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(emoji.emojize("Please type :right_arrow: /start :left_arrow: to start the "
                                                  "getting birds species information \n"
                                                  ))


async def image_data_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # await update.message.reply_text("Please Give the photo please: ")
    # print("hello")
    # photo = await update.message.photo()
    # print("hi")

    await update.message.reply_text(emoji.emojize("Please Give me an Bird or Animal  Image :eagle: :peacock:"))


def handle_response(text: str) -> str:
    text: str = text.lower()
    if 'hello' in text:
        return emoji.emojize("Hey There :smiling_face_with_smiling_eyes: ")

    if 'how are you' in text:
        return emoji.emojize("I am Good :beaming_face_with_smiling_eyes: ")

    if 'animal' in text:
        print("\n\n\n\n I was Inside")
        return 'animal'

    if 'bird' in text:
        return 'bird'
    data = emoji.emojize("Please type :right_arrow: /start :left_arrow: to start the BOT :eagle: "
                         "\nPlease type :right_arrow: /image_data :left_arrow: to start the getting species "
                         "   information "
                         "\n\n\nPlease type :right_arrow: animal :left_arrow: for classification of Animal species"
                         "\n Please type :right_arrow: bird :left_arrow: for classification of Bird Species")

    return data


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)

    else:
        response = handle_response(text)
    new_response = response
    if new_response == 'animal':
        global flag
        flag = 0
        print('BOT', new_response)
        await update.message.reply_text(emoji.emojize("You have chosen Animal :lion: class for the Recognition: "))
    if new_response == 'bird':
        flag = 1
        print('BOT', new_response)
        await update.message.reply_text(emoji.emojize("You have chosen Bird :eagle: class for the Recognition: "))
    print('BOT', new_response)
    await update.message.reply_text(new_response)


def answer(image):
    loaded_model = tf.keras.models.load_model("Third_model.h5")
    loaded_model.summary()
    birds = pd.read_csv("birds.csv")
    birds = birds[birds["data set"] == "train"]
    class_names = birds["labels"].sort_values().drop_duplicates()
    class_names = class_names.to_numpy()
    class_names[380] = 'PARAKETT  AUKLET'
    bird_pred_prob = loaded_model.predict(image)
    bird_pred = tf.round(tf.argmax(bird_pred_prob, axis=1))
    accuracy = tf.reduce_max(bird_pred_prob, axis=1)*100
    ans = class_names[bird_pred[0]]
    scien_name = birds[birds["labels"] == class_names[bird_pred[0]]]["scientific name"].head(1).to_string(index=False)
    return [ans, scien_name , accuracy]


def answer_animal(image):
    animal_model = tf.keras.models.load_model("animal_second_model.h5")
    animal_model.summary()
    fileObject = open("translation.json", "r")
    jsonContent = fileObject.read()
    Dict = json.loads(jsonContent)
    myKeys = list(Dict.keys())
    myKeys.sort()
    sorted_dict = {i: Dict[i] for i in myKeys}
    class_names_animal = []
    for keys in sorted_dict:
        class_names_animal.append(keys)
    labels = []
    for keys in class_names_animal:
        labels.append(sorted_dict[keys])
    labels = np.array(labels)

    animal_pred_prob = animal_model.predict(image)
    animal_pred = tf.round(tf.argmax(animal_pred_prob, axis=1))
    acc = tf.reduce_max(animal_pred_prob, axis=1)*100
    res = labels[animal_pred[0]]
    sci = class_names_animal[animal_pred[0]]
    return [res, sci, acc]


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please wait for sometime ..... We are Processing your request ....")
    photo = await context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(await photo.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype='uint8')

    imge = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    imge = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imge, (224, 224), interpolation=cv2.INTER_AREA)
    plt.imsave("user_photo.png", imge)
    img = tf.constant(img)
    img = tf.expand_dims(img, axis=0)
    global flag
    if flag == 0:
        print("i am animal")
        result = answer_animal(img)

    elif flag == 1:
        print("i am bird")
        result = answer(img)

    else:
        print("i am no one")
        result1 = answer_animal(img)
        result2 = answer(img)
        if result1[2] > result2[2]:
            result = result1

        else:
            result = result2

    flag = 3
    print(result)
    await update.message.reply_text("Here is the photo you sent")
    await update.message.reply_photo('user_photo.png')
    os.remove('user_photo.png')
    await update.message.reply_text(f"The Image is : {result[0]} ")
    await update.message.reply_text(f"Scientific Name : {result[1]}")
    await update.message.reply_text(f"accuracy : {result[2]}")
    await update.message.reply_text(emoji.emojize("Please use :right_arrow: /start :left_arrow: to again ask for image "
                                                  "recognition"))
    await update.message.reply_text(emoji.emojize("Thank You for using our bot :smiling_face_with_smiling_eyes:   "
                                                  ":smiling_face_with_smiling_eyes:"
                                                  ":smiling_face_with_smiling_eyes:"))


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f' Update {update} caused {context.error}')


if __name__ == '__main__':
    print('starting Bot')
    app = Application.builder().token(BOT_TOKEN).build()

    # Commands

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('image_data', image_data_command))

    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    # errors
    app.add_error_handler(error)
    print('Polling')
    app.run_polling(poll_interval=3)


