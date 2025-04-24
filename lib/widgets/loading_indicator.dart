import 'package:flutter/material.dart';
import 'package:loading_animation_widget/loading_animation_widget.dart';

class ThreeDotLoading extends StatelessWidget {
  final Color color;
  final double size;

  const ThreeDotLoading({
    super.key,
    this.color = Colors.blue,
    this.size = 50,
  });

  @override
  Widget build(BuildContext context) {
    return LoadingAnimationWidget.threeArchedCircle(
      color: color,
      size: size,
    );
  }
}
