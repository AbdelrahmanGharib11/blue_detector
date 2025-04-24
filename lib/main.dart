import 'package:blue_detector/screen/homepage.dart';
import 'package:blue_detector/screen/imageviewer.dart';
import 'package:blue_detector/screen/resultscreen.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:blue_detector/services/imageprovider.dart'
    as app_image_provider;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => app_image_provider.ImageProvider(),
      child: MaterialApp(
        debugShowCheckedModeBanner: false,
        home: HomePage(),
        routes: {
          'imageviewer': (_) => const ImageViewer(),
          'resultscreen': (_) => const ResultScreen(),
        },
      ),
    );
  }
}

// Future<void> _processImage(BuildContext context, File imageFile) async {
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

