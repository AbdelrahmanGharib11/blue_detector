import 'dart:io';
import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context) {
    var screenHeight = MediaQuery.sizeOf(context).height;
    final File imageFile = ModalRoute.of(context)!.settings.arguments as File;

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
          'Results',
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
                  'Detected Image',
                  style: TextStyle(
                    color:
                        const Color.fromARGB(255, 7, 61, 123).withOpacity(0.8),
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.left,
                ),
                SizedBox(height: screenHeight * 0.02),
                Container(
                  width: double.infinity,
                  height: screenHeight * 0.4,
                  clipBehavior: Clip.hardEdge,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Image.file(
                    imageFile,
                    fit: BoxFit.cover,
                  ),
                ),
                SizedBox(height: screenHeight * 0.1),
                Text(
                  'Detected: Face',
                  style: TextStyle(
                    color:
                        const Color.fromARGB(255, 7, 61, 123).withOpacity(0.8),
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.left,
                ),
                SizedBox(height: screenHeight * 0.02),
                Text(
                  'Eye Color: Brown/Red',
                  style: TextStyle(
                    color:
                        const Color.fromARGB(255, 7, 61, 123).withOpacity(0.8),
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.left,
                ),
                SizedBox(height: screenHeight * 0.02),
                Text(
                  'Hair Color: Black/Dark',
                  style: TextStyle(
                    color:
                        const Color.fromARGB(255, 7, 61, 123).withOpacity(0.8),
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.left,
                ),
              ],
            ),
          )
        ],
      ),
    );
  }
}
