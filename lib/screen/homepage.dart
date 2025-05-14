import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class HomePage extends StatelessWidget {
  HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    var screenHeight = MediaQuery.sizeOf(context).height;
    return Scaffold(
      backgroundColor: const Color(0xff0B4994),
      body: Stack(
        alignment: Alignment.center,
        children: [
          SizedBox(
            height: screenHeight,
            child: ShaderMask(
              shaderCallback: (Rect bounds) {
                return LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    const Color(0xff0E67D0).withOpacity(0.05),
                    const Color(0xff0E67D0).withOpacity(0.5),
                  ],
                ).createShader(bounds);
              },
              blendMode: BlendMode.srcATop,
              child: Image.asset(
                'asset/image/model1.png',
                fit: BoxFit.contain,
                errorBuilder: (context, error, stackTrace) {
                  return Image.asset("asset/image/failureloading.png",
                      fit: BoxFit.fill,
                      width: double.infinity,
                      height: double.infinity);
                },
              ),
            ),
          ),
          Column(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              const Spacer(flex: 1),
              Text(
                'Add an Image\n to detect what is in',
                textAlign: TextAlign.center,
                style: TextStyle(
                    color:
                        const Color.fromARGB(255, 255, 255, 255).withOpacity(1),
                    fontSize: 30,
                    fontWeight: FontWeight.bold),
              ),
              const Spacer(flex: 10),
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
                      'View Detection Results',
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
              // Row(
              //   mainAxisAlignment: MainAxisAlignment.center,
              //   children: [
              //     ElevatedButton(
              //       onPressed: () => pickImageFromGallery(context),
              //       style: ElevatedButton.styleFrom(
              //         backgroundColor: Colors.white,
              //         iconColor: const Color(0xff0E67D0).withOpacity(0.8),
              //         shape: const CircleBorder(),
              //         padding: const EdgeInsets.all(10),
              //       ),
              //       child: const Icon(
              //         Icons.photo_library_rounded,
              //         size: 40,
              //       ),
              //     ),
              //     const SizedBox(width: 24),
              //     ElevatedButton(
              //       onPressed: () => pickImageFromCamera(context),
              //       style: ElevatedButton.styleFrom(
              //         backgroundColor: Colors.white,
              //         iconColor: const Color(0xff0E67D0).withOpacity(0.8),
              //         shape: const CircleBorder(),
              //         padding: const EdgeInsets.all(10),
              //       ),
              //       child: const Icon(
              //         Icons.camera_alt,
              //         size: 50,
              //       ),
              //     ),
              //   ],
              // ),
              const Spacer(flex: 1),
            ],
          )
        ],
      ),
    );
  }

  final ImagePicker _picker = ImagePicker();

  Future<void> pickImageFromGallery(BuildContext context) async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      File imageFile = File(image.path);
      Navigator.pushNamed(context, 'imageviewer', arguments: imageFile);
    }
  }

  Future<void> pickImageFromCamera(BuildContext context) async {
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      File imageFile = File(image.path);
      Navigator.pushNamed(context, 'imageviewer', arguments: imageFile);
    }
  }
}

  // Future<void> pickImageFromGallery(BuildContext context) async {
  //   final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
  //   if (image != null) {
  //     File imageFile = File(image.path);
  //     Navigator.pushNamed(
  //       context,
  //       'imageviewer',
  //       arguments: imageFile,
  //     );
  //     // await _processImage(context, File(image.path));
  //   }
  // }

  // Future<void> pickImageFromCamera(BuildContext context) async {
  //   final XFile? image = await _picker.pickImage(source: ImageSource.camera);
  //   if (image != null) {
  //     File imageFile = File(image.path);
  //     Navigator.pushNamed(
  //       context,
  //       'imageviewer',
  //       arguments: imageFile,
  //     );
  //     // await _processImage(context, File(image.path));
  //   }
  // }

//   Future<void> _processImage(BuildContext context, File imageFile) async {
//     showDialog(
//       context: context,
//       barrierDismissible: false,
//       builder: (context) => const Center(child: CircularProgressIndicator()),
//     );

//     try {
//       final processedImage = await _backendApi.uploadImage(imageFile);
//       Navigator.of(context).pop();

//       if (processedImage != null) {
//         // Verify the image was actually received
//         final exists = await processedImage.exists();
//         if (exists && processedImage.lengthSync() > 0) {
//           Navigator.pushNamed(
//             context,
//             'imageviewer',
//             arguments: {
//               'originalImage': imageFile,
//               'processedImage': processedImage,
//             },
//           );
//         } else {
//           ScaffoldMessenger.of(context).showSnackBar(
//             const SnackBar(
//                 content: Text('Received empty or invalid image file')),
//           );
//         }
//       } else {
//         ScaffoldMessenger.of(context).showSnackBar(
//           const SnackBar(
//               content: Text('Failed to process image - no data received')),
//         );
//       }
//     } catch (e) {
//       Navigator.of(context).pop();
//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text('Error: ${e.toString()}')),
//       );
//     }
//   }

