import 'package:blue_detector/widgets/loading_indicator.dart';
import 'package:flutter/material.dart';

// ignore: must_be_immutable
class CustomButton extends StatelessWidget {
  CustomButton(
      {super.key,
      required this.screenHeight,
      required this.onTab,
      required this.text,
      required this.isProcessing});

  final double screenHeight;
  bool isProcessing = false;
  String text;
  void Function()? onTab;
  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTab,
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
        child: isProcessing
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
            : Center(
                child: Text(
                  text,
                  style: const TextStyle(
                    color: Color.fromARGB(255, 7, 61, 123),
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
      ),
    );
  }
}
