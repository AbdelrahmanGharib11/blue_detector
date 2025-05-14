import 'package:blue_detector/screen/chooseyouraction.dart';
import 'package:blue_detector/screen/homepage.dart';
import 'package:blue_detector/screen/imageviewer.dart';

import 'package:blue_detector/screen/resultscreen.dart';
import 'package:blue_detector/services/recognizeprovider.dart'
    as app_reco_provider;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:blue_detector/services/imageprovider.dart'
    as app_image_provider;

// Add a global navigator key
final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(
            create: (context) => app_image_provider.ImageProvider()),
        ChangeNotifierProvider(
            create: (context) => app_reco_provider.RecognizeProvider()),
      ],
      child: MaterialApp(
        navigatorKey: navigatorKey,
        debugShowCheckedModeBanner: false,
        home: HomePage(),
        routes: {
          'imageviewer': (_) => const ImageViewer(),
          'resultscreen': (_) => const ResultScreen(),
          'chooseaction': (_) => const ChooseAnActionScreen(),
        },
      ),
    );
  }
}
