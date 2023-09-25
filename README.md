# AirSign: Virtual Signature Capture Through Gesture Recognition

## Abstract

As the world moves towards a paperless future, generating personal, paperless signatures to avoid any paper waste remains a challenge. In this paper, we introduce AirSign, a computer vision-based signature generation system that allows users to generate and save their signatures using gesture recognition. We utilize the MediaPipe framework and transfer learning to train the system to recognize new gestures for AirSign’s signature generation. Our evaluation shows that the system achieves high accuracy in detecting and recognizing new gestures, with a test accuracy of 0.9096. The results demonstrate the effectiveness of the AirSign system in providing a reliable solution for generating signatures using computer vision. With further refinements, the system has the potential to make a significant impact on paperless document signing.

## 1.Introduction

### 1.1 Motivation

As we venture into a paperless future, there are a few things that cannot be easily digitized. One of such things is a signature. Signatures are the representation of one’s identity, and is a crucial piece of acknowledgement on any kind of document. However, with more and more documents embracing the paperless format, the personal signature did not.

The common way to sign a digital document is still to print all the pages out, sign the document, and then scan it back to the computer. The process is not only annoying, complicated, and both time and paper-consuming. Another common digital signature approach is to store one signature as a stamp or digitally generate a signature from one’s name, and reused it in a pdf software whenever you need a signature. However, this approach poses great security issues as each signature is the same, thus easy to be recreated by others and put into fraudulent use.

Therefore, we hope to find a way to minimize the hassle in creating a signature and save paper that serves no purpose but to be a canvas for one to create a signature. We propose to use computer vision to allow users to mimic the signature process in a paperless format. At the same time, we aim to closely emulate the natural signature process so the user can create a signature that is different and authentic each time to ensure security.

### 1.2 Challenges

Generating a signature using computer vision poses several challenges that must be addressed to ensure accurate and reliable results. One challenge is the three-dimensional nature of body and hand movements. Unlike writing on paper, writing in air can be difficult to adjust to, and the user may have a hard time controlling the shape and size of their signature.

In addition, users may also have a hard time adapting to new writing habits. The system must be able to distinguish between intentional hand movements for creating part of the signature and unintentional movements that should not go into part of the signature. For example, the system needs to distinguish when the user is writing, and when the user is only repositioning their hands, mimicking the action of writing and the action of lifting the pen from paper. A sophisticated system is needed for distinguishing between the actions and reacts differently to different kinds of motions.

Finally, the tracking speed is also a concern. Since we want to make the solution lightweight, the user should be able to access and use the service with nothing more than their laptop or phone cameras. However, cameras and phones might have a low resolution and refresh rate, 20 FPS for example, and the system needs to detect and generate the signature even with a low input fidelity. Also, if an machine learning model is used, it needs to be lightweight and reactive enough to keep up with the refresh rate of the device camera input so it can accurately represent the user’s inputs.

Addressing these challenges will be crucial to the development of a reliable and secure signature generation system using computer vision, one that can accurately and intuitively capture the unique aspects of an individual's signature in a paperless format.

### 1.3 Novelty

In this work, we employed Google's MediaPipe framework, specifically its gesture recognition model, to develop a new system for generating signatures using hand movements captured by the user's camera. We used transfer learning and trained the model with an additional 150 epochs on over 2800 photos we captured to recognize four new gestures, each representing an action in the signature process: none (not signing), signing, reset, and save. This customization of the MediaPipe gesture recognition model allowed us to accurately detect and respond to user gestures, providing a natural and intuitive signature generation experience.

To create an end-to-end service, we integrated the gesture recognition model with OpenCV, a popular computer vision library, to capture the user's hand movements in real-time and generate a signature accordingly. The system utilizes MediaPipe's hand landmark detection to track the movement of the user's hand and uses our trained AirSign model to create a signature, which can then be saved to the user's laptop. This integration of MediaPipe and OpenCV allowed us to develop a complete solution for signature generation that is both accurate, lightweight, and user-friendly.

By combining MediaPipe's powerful gesture recognition model with OpenCV's robust computer vision capabilities, we have created a novel system for generating signatures in a paperless and secure manner. The system provides a natural and intuitive user experience, allowing users to create and save their signature easily and conveniently.

### 1.4 Contributions

Our work has made several contributions to the field of signature generation using computer vision. Firstly, we have created a new AirSign model that utilizes transfer learning to the MediaPipe engine, resulting in a highly accurate gesture recognition and signature generation system. By training the model with an additional 150 epochs on over 2800 photos, we were able to customize the MediaPipe gesture recognition model to recognize four new gestures, each representing an action in the signature process, providing users with a natural and intuitive signature creation experience.

In addition, we have also developed a new way of collecting large amounts of picture data for training computer vision models. Since it is not easy to collect a large dataset of gesture images, we used a video-to-picture service to produce a large amount of data quickly. This innovative approach to data collection has the potential to benefit other computer vision applications that require large amounts of data for training, especially in scenarios where collecting such data is challenging or time-consuming.

Overall, our work has contributed to the development of a robust and user-friendly signature generation system using computer vision. Our AirSign model, trained on a large dataset of images collected using our innovative data collection method, has achieved high accuracy and reliability, providing a reliable implementation to a paperless solution for creating signatures.

## 2 Related work

Hand gestures are a form of nonverbal communication that can be used in several fields such as communication between deaf-mute people, robot control, human–computer interaction (HCI), home automation and medical applications. [1] There are many ways we can achieve hand gesture recognition.

### 2.1 Wearable-Sensor-Based Detection

One possible solution for gesture recognition would require the user to put on gloves that are packed with sensors. This method achieves high accuracy in tracking through a combination of flex sensors, gyroscope, and accelerometer on the gloves. The hand movements can be captured through gyroscope and accelerometer so its absolute location would be readily available. The finger movements will be tracked through flex sensors installed on the back of the gloves. Bending the finger would trigger the sensors to record any minute movements. The problem with this type of tracking is the requirement of a carefully engineered gesture-capturing glove. The gloves can be hard to find and very expensive. Only a few manufacturers make these gloves and they can cost upwards of thousands of dollars.
Apart from a wearable solution, depth sensors can also be used to capture hand gestures. A depth sensing camera can provide a 3D geometric information of the hand. Using skeletal joint location, joint orientation, and angles and spaces between joints, the system can calculate the positions of each finger and achieve gesture recognition. A common depth sensor used in this process is the Microsoft Kinect. This approach only requires a simple depth sensor and is a relatively inexpensive solution compared to the one above. However, the requirement of an additional sensor still does not meet the requirement of a lightweight solution.[1 reference from proposal]
Another solution, which is the one that will be explored in this project, is gesture recognition using a computer vision based system. The computer-vision-based approach utilizes a camera to capture the user's hand gestures instead of the glove with sensors. There have been many types of camera used for gesture recognition - including monocular, fisheye, and infrared(IR) cameras. [1] Although camera-based gesture recognition has the advantage of freeing the user from wearing any external device, the method has its own limitations. Hand recognition and isolation, skin color differentiation, depth sensing, lighting variability can all affect the accuracy of the gesture sensing and tracking model.
2.2 Computer-Vision-Based Detection
Apart from using glove and external wearable sensors, another approach is to use computer vision for gesture detection.

A large portion of previous research uses specialized hardware with depth sensing capabilities. For example, Microsoft Kinect, a series of motion sensing devices produced by Microsoft was extensively used in one of the previous studies. Kinect contains a RGB camera, an infrared projector and detector to enable depth sensing capabilities. The benefit of Kinect is it’s readily available with the ability to capture hand motions and body skeletal tracking. However, the Kinect sensor does not work by itself. Like many stationary sensors, it would still need to be connected to a laptop or desktop computer to perform its functions.

An alternative approach is to use on-device cameras, which relies more on the capabilities of software and machine learning models that process the input images and videos. For example, PoseNet is a deep learning-based human pose estimation model that can detect and track skeletal body poses from images or video frames. It uses a convolutional neural network (CNN) to generate a heatmap of body joint locations, which can then be used to estimate the pose of a human body. PoseNet is a popular model for a wide range of applications, such as fitness tracking, sports analysis, and virtual try-on. However, PoseNet is only capable of tracking the wrist joint, which does not match our expectation.

The on-device camera approach is the most promising for accessibility, since no one should be carrying a specialized camera when they are urgently in need of a signature. As long as the machine learning models are lightweight enough, we would be able to implement it on a personal laptop or even on mobile devices.

## 3 Overview

In this section, we provide an overview of the problem related to computer-vision based signature generation. Then in the following section, we provide a high level overview of our proposed AirSign architecture, with its relevant components.

### 3.1 Problem Formulation

We formalize the problem of signature generation using computer vision as follows: Given a live video stream captured by a user's laptop camera, the system should be able to track the movement of the user's hand in real-time, generate a signature based on the movements of the hand, and save the generated signature to the user's laptop.

Let X be the input video stream captured by the camera, and let Y be the output generated signature. The system takes in X and outputs Y, where Y is a 2D image of the user's signature.

The system should be able to address the challenges of generating signatures using computer vision, such as the variability in signature styles and the 3-dimensional nature of writing in air. The system should also incorporate security measures to ensure that the generated signature is unique and cannot be easily replicated.

### 3.2 Solution Overview

To address the challenges of signature generation using computer vision, we developed a novel system called AirSign, which utilizes Google's MediaPipe framework and OpenCV. The AirSign system employs MediaPipe's gesture recognition model, customized with transfer learning on 150 epochs and over 2800 photos, to recognize the gestures for signing, resetting, saving, and none (not-signing). This allows for natural and intuitive signature creation by detecting the user's hand movements in real-time.

The AirSign system integrates the gesture recognition model with OpenCV, which captures the user's hand movements in real-time using the camera input, and tracks the movement of the hand using MediaPipe's hand landmark detection. The system then generates a signature based on the movements of the hand, and saves the generated signature to the user's laptop.

Overall, the AirSign system provides a reliable, secure, and paperless solution for generating signatures using computer vision, with natural and intuitive user experience.

## 4 Model Implementation

The model mainly consists of three building blocks: Google MediaPipe for gesture recognition, OpenCV for capturing and recording, and our development that acts as adhesive to combine everything seamlessly. In this section we will first cover what Google MediaPipe and OpenCV are, then explain how we incorporated them into our system.

### 4.1 Google MediaPipe Engine

MediaPipe is a versatile, open-source, cross-platform framework designed for developing machine learning (ML) applications and perception pipelines, with a primary focus on real-time processing. Developed by Google, it provides a comprehensive ecosystem that allows developers to build, evaluate, and deploy complex ML solutions rapidly. MediaPipe offers a collection of pre-built models, components, and tools to create custom perception pipelines for a wide range of applications, such as hand tracking, face detection, object detection, and more.

MediaPipe is designed to simplify the process of building perception pipelines, which are sequences of ML models and data processing components that transform raw input data (e.g., images, video, or audio) into useful information or actions. The framework is modular, allowing developers to create, test, and iterate on individual components or the entire pipeline, making it easy to integrate new models and algorithms or modify existing ones.

MediaPipe Hands is a state-of-the-art ML solution built on the MediaPipe framework, specifically designed for real-time hand tracking on various devices, including smartphones, AR/VR headsets, and computers. It offers an efficient, robust, and flexible pipeline that enables developers to integrate hand tracking into their applications, paving the way for gesture recognition, augmented reality, human-computer interaction, and many other use cases.

The pipeline consists of two main components: a palm detector and a hand landmark model. The palm detector is responsible for identifying the presence and location of a hand in an image or video frame, while the hand landmark model predicts the 3D coordinates of 21 hand keypoints, representing the hand's pose and anatomical structure.

#### 4.1.1 Palm Detector

The palm detector is based on a neural network architecture called BlazePalm, which is optimized for real-time performance on mobile devices. BlazePalm is a lightweight, single-shot detector with a two-step pipeline that consists of a region proposal network (RPN) and a subsequent classification and regression network. The RPN generates a set of candidate bounding boxes, and the classification and regression network refines the box coordinates and scores the candidates.

To improve the palm detector's accuracy and efficiency, the authors employed several techniques, such as anchor box scaling, data augmentation, and multi-task learning. Anchor box scaling helps the model detect hands of various sizes by using anchor boxes at multiple scales, while data augmentation enhances the training data by applying random transformations, such as rotation, scaling, and flipping. Multi-task learning combines the classification and regression tasks into a single network, allowing the model to learn shared features more effectively.

#### 4.1.2 Hand Landmark Model

The hand landmark model takes the cropped image of a detected hand (obtained from the palm detector) as input and predicts the 3D coordinates of 21 hand keypoints. These keypoints represent the hand's anatomical structure, including fingertips, knuckles, and the wrist. The model's architecture is not explicitly detailed in the paper, but it is based on a neural network that can efficiently predict hand keypoints in real-time.

The hand landmark model is trained on a diverse dataset containing a variety of hand poses, orientations, and sizes, as well as different lighting conditions and occlusions. This ensures that the model is robust and can handle various challenging scenarios in real-world applications.

#### 4.1.3 Gesture Recognition

The hand gesture recognition model in MediaPipe uses hand landmarks produced by the MediaPipe Hands model to classify a hand pose into one of eight gesture classes: Closed Fist, Open Palm, Pointing Up, Thumb Down, Thumb Up, Victory, I Love You, and None of the above gestures. The solution architecture consists of a two-step neural network pipeline that includes an embedding model followed by a classification model. This pipeline operates on hand landmarks and related information for a single hand but does not directly process any images (i.e., RGB pixel data).

The pipeline consumes the outputs of the MediaPipe Hands model, including 21 3-dimensional screen landmarks represented as a 1x63 tensor and normalized by image size, a float scalar representing the handedness probability of the predicted hand, and 21 3-dimensional metric scale world landmarks represented as a 1x63 tensor and normalized by image size. No image data is directly input into the model.

The output of the pipeline is an 8-element vector that predicts the probability of each of the following classes: 0th-element corresponds to the probability that the hand pose is not a known hand gesture to the model, and the 1st-7th elements correspond to the probability of the hand pose being one of the seven known gestures.

The embedding model is a fully connected neural network with residual blocks and a regression model architecture. It takes as input the 21 3-dimensional screen landmarks, handedness probability, and the 21 3-dimensional metric scale world landmarks. The output of the embedding model is a float tensor 128x1 embedding tensor representing the hand landmarks, which is further used in the classification model head, described in the next section.

The classification model is a fully connected neural network with a classification model architecture. It takes as input the 128x1 embedding tensor representing the hand landmarks and outputs an 8-element vector that predicts the probability for each of the eight aforementioned gesture classes.

Taking advantage of the robust hand landmark detection model behind the MediaPipe hand gesture recognition model, we were able to freeze the landmark layers and only retrain the recognition layers. This allows a straightforward training set acquisition, as we nolonger need to label the land skeleton of each image, and instead only need to label each image with their corresponding label.

### 4.2 OpenCV

OpenCV (Open Source Computer Vision Library) is a popular open-source computer vision and machine learning software library. It provides a set of powerful algorithms and functions that enable developers to perform a wide range of computer vision tasks, such as image processing, object detection, face recognition, and gesture recognition.

The library was first released in 2000 by Intel Corporation and has since been adopted by a wide range of developers and organizations for various computer vision applications. OpenCV is written in C++, but it also provides interfaces for several other programming languages, such as Python, Java, and MATLAB.

OpenCV provides a vast array of functions and tools for image and video processing. These functions range from basic operations such as thresholding and filtering, to more complex tasks such as object detection, image segmentation, and feature detection. OpenCV also provides a range of tools for machine learning, including support for various popular machine learning frameworks, such as TensorFlow and PyTorch, which is the primary reason for including the library in AirSign.

One of the key features of OpenCV is its support for real-time computer vision applications, perfectly suitable for our need of real-time hand capturing. OpenCV provides interfaces for several popular camera APIs, such as DirectShow, Video4Linux, and AVFoundation, allowing developers to easily capture and process video streams in real-time. OpenCV also supports parallel processing, allowing developers to take advantage of multi-core processors to improve performance. We will later explore the

OpenCV is a versatile and powerful library that has been used in a wide range of computer vision applications, from industrial automation to autonomous vehicles. Its robustness, flexibility, and extensive documentation make it a popular choice for developers and researchers in the field of computer vision.

In our work, we utilized OpenCV to capture the user's hand movements in real-time and integrate it with MediaPipe's hand landmark detection for signature generation. This integration allowed us to create a robust and accurate signature generation system that is both natural and intuitive to use.

## 5 Evaluation

In this section, we provide a comprehensive review of our evaluation of AirSign. In section 5.1, we first talk about the experiment configuration regarding data processing and training. Then in section 5.2, we mention the difficulties encountered in defining baselines for AirSign. Lastly, section 5.3 provides an extensive review of the model performance and our study with 10 college undergraduates participants.

### 5.1 Evaluation setup:

In order to evaluate the performance of the AirSign system, we utilized a dataset of hand movement sequences captured using three cameras: a M1 Macbook Pro’s laptop camera, an Intel Macbook Pro’s(2019) laptop camera, and a ANVASKTEK 1080P desktop webcam. The dataset was randomly split into 80% training data, 10% validation data, and 10% testing data to ensure that the model was trained and evaluated on diverse data samples.

To facilitate collaboration within the team, we used Google Colaboratory(Google Colab) platform. Google Colab is a cloud-based platform that provides a free environment for running Python code, including machine learning and deep learning models. We utilized Google Colab for both training and evaluation purposes to allow more efficient communication, and to take advantage of MediaPipe's readily available models.

According to Google Colab, the machine that our model trained on is using an NVIDIA A100 Graphics Card.

### 5.2 Baselines, Metrics

One baseline AirSign uses is the difference in amount of time it takes to complete a signature. With a machine generating the signature for the user, it would take slightly longer due to the absence of a physical pen. For example, when a user is writing a name “William”, you would need to spend extra time dotting the dots even when writing in cursive. Compared to the action sequence of “gesture to stop, find dot location, gesture to write dot, gesture to stop,” a pen performs the job much faster.

However, we concluded that the quality of the signature cannot simply be quantified. It would be very hard to come up with a quantitative method to assign a signature a grade, since everyone’s signature is different and distinct. The ability to generate a distinct, personal signature is the most important reason for developing a system such as AirSign.

Signatures are inherently subjective and vary greatly depending on individual style, speed, and although unaccounted for in AirSign, pressure, among other factors. This makes it challenging to establish a standardized baseline for signature generation using computer vision.

Additionally, there is a lack of established benchmarks or standardized datasets for signature generation using computer vision. While there are some publicly available datasets for signature verification or recognition, these datasets are often limited in scope and may not provide sufficient coverage of the variations and nuances present in natural signature generation.

To address a first-hand signature verification by the user, we have added the “reset” gesture, so the user can scrap the signature generated so far in the system and start fresh. The user may repeat the process for as many times as they like, until they have signed the signature that meet their standard and expectation.

While it may be difficult to establish direct baselines and metrics for assessing the AirSign system, our evaluation setup provides a robust framework for assessing the effectiveness of the system in generating signatures using computer vision.

### 5.3 Performance

To train the customized gesture recognition model, we utilized transfer learning using the model that MediaPipe has provided. MediaPipe's model has the capability to recognize hand landmarks and skeleton with high accuracy, which allowed us to train the model to detect new gestures without worrying about the hand landmark input inaccuracy.

To optimize the model's performance, we used the Focal loss function, which was determined to be the best performing loss function for gesture recognition by MediaPipe's paper. The Focal loss function allowed the model to focus on difficult-to-classify examples, resulting in better accuracy. To further optimize the model's performance, we utilized the Adam optimizer, which is a popular optimizer in deep learning due to its ability to converge quickly and efficiently.

After training, the model achieved a test loss of 0.1329 and a test accuracy of 0.9096. These results demonstrate the effectiveness of the customized gesture recognition model in accurately detecting and recognizing the new gestures for signature generation.

To further evaluate the performance of the AirSign system, we conducted a user study with 10 college undergraduate participants. The study involved participants using the AirSign system to generate their signature, and providing feedback on the ease of use, naturalness, and security of the system. The feedback was quite positive, with participants noting that the system was intuitive with signatures generated that are natural and unique to each individual. However, the process of generating signatures can take some practice to get used to.

Many of the participants intuitively start writing their signature at the center of the canvas instead of on one end of the canvas, like one would usually do if the participant is signing on a piece of paper. This is primarily due to the fact that the laptop camera is placed right in front of the participant. Many of them have not grown the spatial awareness at the beginning of the study and signed their signatures starting at the center only to find that they are running out of space very quickly. The lacking of an indicator of where one should start signing is a common feedback provided by the participants.

Another issue the participants faced is related to the depth of the camera. Since the participants are signing in three-dimensional space, to sign a good signature, one must envision a imaginary two-dimensional plane in front of them that they are signing on. Otherwise, excessive movement of hands in the direction vertical to the camera might lead to different characters of the signature being signed in different sizes, as the laptop camera does not have depth sensing capabilities and cannot distinguish between sizes.

In addition, other feedback includes difficulties to signing dots, mis-recognition of “sign” gesture while facing towards the camera, and the mis-categorization of “sign” gesture as “save.” The last feedback was quickly amended with a longer dead period that’s required before AirSign closes and saves the signature.

Overall, our evaluation setup utilized state-of-the-art techniques in transfer learning and loss function selection to create a customized gesture recognition model that accurately detects new gestures for signature generation.

## Limitations and Discussions

### 6.1 Limitations

While the AirSign system has demonstrated high accuracy in detecting and recognizing gestures directing the signature generation process, there are a few limitations that we would aim to address in future works.

One limitation is the accuracy of our gesture recognition model. Although achieving over 90% accuracy in correct categorization is sufficient for our current stage, an improvement in accuracy would directly translate to better user experience. For example, the current model may not recognize the hand in certain angles, such as when the user points their hand forward into the camera. This can lead to unintentional or missing strokes on the screen where the user would have to start over in the signing process.

Another limitation is the current process for exporting the generated signature. While the current system allows users to generate and save their signature on their laptop, a more seamless process would be to export the signature directly onto a PDF document that awaits the signature. This would allow for a more intuitive and streamlined signature generation process.

Finally, another limitation is the current method for ending strokes. While the current system allows users to end strokes using a non-sign gesture and restart with a sign gesture, a more intuitive experience would be to incorporate depth estimation to allow users to "lift up" from the virtual plane to end a stroke.

### 6.2 Discussions

Compared to our original proposal, we are satisfied with the results of the AirSign system. We were able to achieve our original goal of building an interactive and useful gesture recognition signature tool that can be used for paperless document signing.

Beyond the immediate impact of our work on paperless document signing, there are many potential broader impacts. For example, the AirSign system could potentially help to push our society one step closer to full paperless operations, reducing the use of paper and other resources in daily operations. Additionally, the system could potentially help to secure the signature process for millions of documents signed every day, improving the overall security and trustworthiness of signed documents.

Overall, the AirSign system demonstrates the effectiveness of utilizing state-of-the-art techniques in transfer learning to create a customized gesture recognition model that accurately detects new gestures for signature generation. Future works could address the limitations discussed above and further improve the user experience and broader impact of the system.

## 7 Conclusion

In this paper, we presented AirSign, a computer vision-based signature generation system that allows users to generate and save their signatures using gesture recognition. We utilized the MediaPipe framework and specifically, the gesture recognition model, to train the system to recognize four new gestures - none (not-signing), signing, reset, and save - representing actions in the signature process.

Our evaluation setup utilized state-of-the-art models in transfer learning to create a customized gesture recognition model that accurately detects new gestures for AirSign’s signature generation. The high accuracy of the model, combined with the positive feedback from the user study, demonstrates the effectiveness of the AirSign system in providing a reliable solution for generating paperless signatures using computer vision.

While there are limitations that we aim to address in future works, such as improving the accuracy of the gesture recognition model and incorporating depth estimation for ease of use, the AirSign system shows great potential for reducing unnecessary paper waste in daily operations and allowing for accessible signature generation anywhere, anytime.
