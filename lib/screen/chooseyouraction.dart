import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:blue_detector/screen/imageviewer.dart';
import 'package:blue_detector/main.dart';

class ChooseAnActionScreen extends StatelessWidget {
  const ChooseAnActionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    var screenHeight = MediaQuery.sizeOf(context).height;

    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xff0E67D0).withOpacity(0.8),
        leading: IconButton(
          onPressed: () => Navigator.pop(context),
          icon: Icon(
            Icons.arrow_back_ios_new_rounded,
            color: const Color.fromARGB(255, 7, 61, 123).withOpacity(0.8),
          ),
        ),
        title: Text(
          'Blue Detector',
          style: TextStyle(
            color: const Color.fromARGB(255, 7, 61, 123).withOpacity(0.8),
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
      ),
      body: Stack(
        alignment: Alignment.center,
        children: [
          Container(
            height: screenHeight,
            width: double.infinity,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  const Color(0xff0E67D0).withOpacity(0.8),
                  const Color(0xff0E67D0).withOpacity(0.3)
                ],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'Choose an Action',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                      color: const Color.fromARGB(255, 255, 255, 255)
                          .withOpacity(1),
                      fontSize: 45,
                      fontWeight: FontWeight.bold),
                ),
                const SizedBox(
                  height: 60,
                ),
                InkWell(
                  onTap: () {
                    showTwoButtonDialog(context: context);
                  },
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 20),
                    width: double.infinity,
                    height: screenHeight * 0.1,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(16),
                      gradient: LinearGradient(
                        colors: [
                          const Color.fromARGB(255, 255, 255, 255)
                              .withOpacity(0.5),
                          const Color.fromARGB(255, 120, 159, 203)
                              .withOpacity(0.8)
                        ],
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                      ),
                    ),
                    child: const Center(
                      child: Text(
                        'Face Detection',
                        style: TextStyle(
                          color: Color.fromARGB(255, 5, 76, 156),
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                ),
                const SizedBox(
                  height: 40,
                ),
                InkWell(
                  onTap: () {
                    Navigator.pushNamed(context, 'chooseaction');
                  },
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 20),
                    width: double.infinity,
                    height: screenHeight * 0.1,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(16),
                      gradient: LinearGradient(
                        colors: [
                          const Color.fromARGB(255, 255, 255, 255)
                              .withOpacity(0.5),
                          const Color.fromARGB(255, 120, 159, 203)
                              .withOpacity(0.8)
                        ],
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                      ),
                    ),
                    child: const Center(
                      child: Text(
                        'Add Another Dataset',
                        style: TextStyle(
                          color: Color.fromARGB(255, 5, 76, 156),
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          )
        ],
      ),
    );
  }
}

final ImagePicker _picker = ImagePicker();

Future<void> pickImageFromGallery(BuildContext context) async {
  try {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      File imageFile = File(image.path);
      navigatorKey.currentState?.push(
        MaterialPageRoute(
          builder: (context) => ImageViewer(imageFile: imageFile),
        ),
      );
    }
  } catch (e) {
    print('Error picking image from gallery: $e');
  }
}

Future<void> pickImageFromCamera(BuildContext context) async {
  try {
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);

    if (image != null) {
      File imageFile = File(image.path);
      navigatorKey.currentState?.push(
        MaterialPageRoute(
          builder: (context) => ImageViewer(imageFile: imageFile),
        ),
      );
    }
  } catch (e) {
    print('Error picking image from camera: $e');
  }
}

void showTwoButtonDialog({
  required BuildContext context,
}) {
  showDialog(
    context: context,
    builder: (BuildContext context) {
      return AlertDialog(
        title: const Text(
          'Choose an Option!',
          style: TextStyle(
            color: Colors.white,
          ),
        ),
        backgroundColor: const Color.fromARGB(255, 17, 80, 255),
        actionsPadding:
            const EdgeInsets.symmetric(horizontal: 10, vertical: 20),
        buttonPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 20),
        actionsAlignment: MainAxisAlignment.spaceAround,
        actions: <Widget>[
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.white,
            ),
            child: const Text(
              'Gallery',
              style: TextStyle(
                color: Color.fromARGB(255, 17, 80, 255),
                fontSize: 16,
              ),
            ),
            onPressed: () {
              Navigator.of(context).pop();
              pickImageFromGallery(context);
            },
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.white,
            ),
            child: const Text(
              'Camera',
              style: TextStyle(
                color: Color.fromARGB(255, 17, 80, 255),
                fontSize: 16,
              ),
            ),
            onPressed: () {
              Navigator.of(context).pop();
              pickImageFromCamera(context);
            },
          ),
        ],
      );
    },
  );
}
