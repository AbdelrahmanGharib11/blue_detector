import 'dart:io';
import 'package:blue_detector/services/imageprovider.dart'
    as app_image_provider;
import 'package:blue_detector/services/recognizeprovider.dart'
    as app_reco_provider;
import 'package:blue_detector/widgets/custombutton.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class ImageViewer extends StatelessWidget {
  final File? imageFile;

  const ImageViewer({super.key, this.imageFile});

  @override
  Widget build(BuildContext context) {
    int key = 0;
    var screenHeight = MediaQuery.sizeOf(context).height;
    final recoProvider = Provider.of<app_reco_provider.RecognizeProvider>(
      context,
    );
    final imageProvider =
        Provider.of<app_image_provider.ImageProvider>(context);
    final File displayImage = imageFile ??
        (ModalRoute.of(context)?.settings.arguments as File? ?? File(''));

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
              children: [
                Container(
                  width: double.infinity,
                  height: screenHeight * 0.52,
                  clipBehavior: Clip.hardEdge,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Image.file(
                    displayImage,
                    fit: BoxFit.fill,
                  ),
                ),
                SizedBox(height: screenHeight * 0.1),
                CustomButton(
                  screenHeight: screenHeight,
                  text: 'View Detection Results',
                  isProcessing: imageProvider.isProcessing,
                  onTab: () async {
                    key = 0;
                    final imageProvider =
                        Provider.of<app_image_provider.ImageProvider>(context,
                            listen: false);
                    await imageProvider.processImage(displayImage);
                    if (imageProvider.processedImage != null) {
                      Navigator.pushNamed(context, 'resultscreen',
                          arguments: key);
                    }
                  },
                ),
                const SizedBox(
                  height: 16,
                ),
                CustomButton(
                    screenHeight: screenHeight,
                    text: 'View Recognition Results',
                    onTab: () async {
                      key = 1;

                      final recoProvider =
                          Provider.of<app_reco_provider.RecognizeProvider>(
                              context,
                              listen: false);
                      await recoProvider.recognizeFace(displayImage);
                      if (recoProvider.recognitionResults != null) {
                        Navigator.pushNamed(context, 'resultscreen',
                            arguments: key);
                      }
                    },
                    isProcessing: recoProvider.isLoading
                    // Provider.of<app_reco_provider.RecognizeProvider>(context,
                    //         listen: false)
                    //     .isLoading,
                    ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
