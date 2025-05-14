import 'dart:io';
import 'package:flutter/material.dart';
import 'package:blue_detector/services/backend_api.dart';

class RecognizeProvider with ChangeNotifier {
  // State variables
  bool _isLoading = false;
  String? _error;
  Map<String, dynamic>? _recognitionResults;
  bool _serviceHealthy = false;

  bool get isLoading => _isLoading;
  String? get error => _error;
  Map<String, dynamic>? get recognitionResults => _recognitionResults;
  bool get serviceHealthy => _serviceHealthy;

  final BackendApi _backendApi = BackendApi();

  Future<void> checkServiceHealth() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _serviceHealthy = await _backendApi.checkFaceRecognitionHealth();
    } catch (e) {
      _error = "Health check failed: ${e.toString()}";
      _serviceHealthy = false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> recognizeFace(File imageFile) async {
    _isLoading = true;
    _error = null;
    _recognitionResults = null;
    notifyListeners();

    try {
      final results = await _backendApi.recognizeFace(imageFile);
      if (results != null) {
        _recognitionResults =
            await _backendApi.processRecognitionResults(results);
      } else {
        _error = "Recognition failed - no results returned";
      }
    } catch (e) {
      _error = "Recognition error: ${e.toString()}";
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<File?> getResultFile(String filename) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      return await _backendApi.getResultFile(filename);
    } catch (e) {
      _error = "Error getting result file: ${e.toString()}";
      return null;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }

  void clearResults() {
    _recognitionResults = null;
    _error = null;
    notifyListeners();
  }
}
