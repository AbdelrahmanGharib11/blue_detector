import 'dart:io';
import 'package:blue_detector/widgets/loading_indicator.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:blue_detector/services/imageprovider.dart'
    as app_image_provider;

class ImageViewer extends StatelessWidget {
  const ImageViewer({super.key});

  @override
  Widget build(BuildContext context) {
    var screenHeight = MediaQuery.sizeOf(context).height;
    final imagefile = ModalRoute.of(context)!.settings.arguments as File;
    final imageProvider =
        Provider.of<app_image_provider.ImageProvider>(context);

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
                Container(
                    height: screenHeight * 0.4,
                    clipBehavior: Clip.antiAlias,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Image.file(
                      imagefile,
                      fit: BoxFit.cover,
                    )),
                SizedBox(height: screenHeight * 0.15),
                InkWell(
                  onTap: () async {
                    final imageProvider =
                        Provider.of<app_image_provider.ImageProvider>(context,
                            listen: false);
                    await imageProvider.processImage(imagefile);
                    if (imageProvider.processedImage != null) {
                      Navigator.pushNamed(context, 'resultscreen');
                    }
                  },
                  child: Container(
                    width: double.infinity,
                    height: screenHeight * 0.1,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(16),
                      gradient: LinearGradient(
                        colors: [
                          const Color(0xff0E67D0).withOpacity(0.5),
                          const Color(0xff0E67D0).withOpacity(0.8)
                        ],
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                      ),
                    ),
                    child: imageProvider.isProcessing
                        ? const Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                'Loading',
                                style: TextStyle(
                                  color: Color.fromARGB(255, 7, 61, 123),
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                ),
                                textAlign: TextAlign.center,
                              ),
                              SizedBox(width: 10),
                              ThreeDotLoading(
                                color: Color.fromARGB(255, 7, 61, 123),
                                size: 24,
                              ),
                            ],
                          )
                        : const Center(
                            child: Text(
                              'View Detection Results',
                              style: TextStyle(
                                color: Color.fromARGB(255, 7, 61, 123),
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                              ),
                              textAlign: TextAlign.center,
                            ),
                          ),
                  ),
                )
              ],
            ),
          )
        ],
      ),
    );
  }
}
