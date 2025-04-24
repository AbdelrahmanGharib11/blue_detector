import 'dart:io';
import 'package:flutter/material.dart';
import 'package:blue_detector/services/backend_api.dart';

class ImageProvider with ChangeNotifier {
  File? _originalImage;
  File? _processedImage;
  bool _isProcessing = false;
  String _detectionResults = '';

  File? get originalImage => _originalImage;
  File? get processedImage => _processedImage;
  bool get isProcessing => _isProcessing;
  String get detectionResults => _detectionResults;

  final BackendApi _backendApi = BackendApi();

  Future<void> processImage(File imageFile) async {
    _originalImage = imageFile;
    _isProcessing = true;
    notifyListeners();
    print('lets start');
    try {
      _processedImage = await _backendApi.uploadImage(imageFile);
      if (_processedImage != null) {
        
        _detectionResults =
            "Face detected"; 
      } else {
        _detectionResults = "Detection failed";
      }
    } catch (e) {
      _detectionResults = "Error: ${e.toString()}";
    } finally {
      _isProcessing = false;
      notifyListeners();
    }
  }

  void clearImages() {
    _originalImage = null;
    _processedImage = null;
    _detectionResults = '';
    notifyListeners();
  }
}
