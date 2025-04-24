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

