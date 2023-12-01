import torch
import cv2
import numpy as np
import datetime
import os
import shutil
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def classification(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Compute the classification report
    cr = classification_report(y_true, y_pred)
    print("Classification Report:\n", cr)



    # Compute the ROC AUC score
    auc_score = roc_auc_score(y_true, y_pred)

    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')

def detect_weld_cut(image, model):
    start_time = datetime.datetime.now()
    results = model(image)

    person = []
    hat = []
    finalStatus = ""
    Saw = False
    Fire_Extinguisher= False
    Fire_Prevention_Net = False


    if np.shape(results.xyxy[0].cpu().numpy())[0] > 0:
        for (x0, y0, x1, y1, confi, clas) in results.xyxy[0].cpu().numpy():
            if confi > 0.5:
                # print(x0, y0, x1, y1, confi, clas)
                box = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                box2 = [int(x0), int(y0), int(x1), int(y1)]
                #box3 = [int(x0), int(y0), int(x1), int(y1)]
                if int(clas) == 0 :
                    cv2.rectangle(image, box, (0, 130, 0), 2)
                    cv2.putText(image, "Saw {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 200, 0), 2)
                    Saw = True

                elif int(clas) == 1 :
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "Fire Extinguisher {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                    Fire_Extinguisher = True
                elif int(clas) == 2:
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "Fire Prevention Net {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                    Fire_Prevention_Net = True

                elif int(clas) == 3:
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "Hard Hat {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                    hat.append(box2)  # Add the box coordinates to the person array
                elif int(clas) == 4 :
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "Welding Equipment {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                    #Welding_Equipment = True


                elif int(clas) == 5:

                    person.append(box2)  # Add the box coordinates to the person array

        for perBox in person:
            hatDetected = False

            for hatBox in hat:
                if int((hatBox[0] + hatBox[2] )/2) > int(perBox[0]) and int((hatBox[0] + hatBox[2] )/2) < int(perBox[2]):
                    if hatBox[1] >= perBox[1] - 20:
                        hatDetected = True

            if hatDetected:
                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 180, 0), 2)
                cv2.putText(image, "Worker with Hardhat", (perBox[0], perBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 180, 0), 2)
                if finalStatus != "UnSafe":
                    finalStatus = "Safe"

            else:
                finalStatus = "UnSafe"
                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 0, 255), 2)
                cv2.putText(image, "Worker without HardHat", (perBox[0], perBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
        if Fire_Extinguisher:
            if finalStatus != "UnSafe":
                finalStatus = "Safe"
            cv2.putText(image, "Fire Extinguisher: Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 3)

        else:
            finalStatus = "UnSafe"
            cv2.putText(image, "Fire Extinguisher: Missing", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)
        if Saw == True:
            if Fire_Prevention_Net == True:

                if finalStatus != "UnSafe":
                    finalStatus = "Safe"
                cv2.putText(image, "Cutting: Fire Prevention Net detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 3)
            else:

                finalStatus = "UnSafe"
                cv2.putText(image, "Cutting: Fire Prevention Net Missing", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)
            # if Fire_Extinguisher == False and len(person) < 1:
            #     finalStatus = "UnSafe"
            #     cv2.putText(image, "Cutting: Fire Extinguisher Missing", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
            #                 1, (0, 0, 255), 3)

        if finalStatus != "Safe":
            cv2.putText(image, f"{finalStatus} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(image, f"{finalStatus} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)



    end_time = datetime.datetime.now()
    inference_time = (end_time - start_time).total_seconds()
    #print(f"Inference Time: {inference_time} seconds")

    if finalStatus == "UnSafe":
        return image, 0, inference_time
    else:
        return image, 1, inference_time




if __name__ == "__main__":

    total_inference_time = 0.0
    ground_truth = []
    prediction = []

    weld_cut_model = torch.hub.load('WongKinYiu/yolov7', 'custom', './Cutting_Welding.pt', force_reload=False)

    print("Loading...")



    safe_folder_path = "./Safe"
    unsafe_folder_path = "./Unsafe"
    output_folder_path = "./Predictions"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Folder '{output_folder_path}' created.")
    else:
        shutil.rmtree(output_folder_path)
        os.makedirs(output_folder_path)
        print(f"Old folder deleted and recreate the folder '{output_folder_path}' .")

    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(safe_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(safe_folder_path, image_file)
        image = cv2.imread(image_path)
        ground_truth.append(1)
        mobile_scaff_result_image, status, inference_time = detect_weld_cut(image, weld_cut_model)
        total_inference_time += inference_time
        prediction.append(status)
        output_path = os.path.join(output_folder_path, image_file  + ".jpg")
        cv2.imwrite(output_path, mobile_scaff_result_image)

    # Get a list of image file names in the folder
    unsafe_image_files = [f for f in os.listdir(unsafe_folder_path) if
                          f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for unsafe_image_file in unsafe_image_files:
        # Read the image
        image_path = os.path.join(unsafe_folder_path, unsafe_image_file)
        image = cv2.imread(image_path)
        ground_truth.append(0)
        mobile_scaff_result_image, status, inference_time = detect_weld_cut(image, weld_cut_model)
        total_inference_time += inference_time
        prediction.append(status)
        output_path = os.path.join(output_folder_path, unsafe_image_file + ".jpg")
        cv2.imwrite(output_path, mobile_scaff_result_image)

    num_images = len(image_files) + len(unsafe_image_files)
    avg_inference_time = total_inference_time / num_images if num_images > 0 else 0.0
    print(f"Average Inference Time: {avg_inference_time} seconds")
    print("FPS = ", 1/avg_inference_time)
    classification(ground_truth, prediction)

    cv2.waitKey(0)
