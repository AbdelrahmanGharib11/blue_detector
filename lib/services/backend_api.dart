import 'dart:io';
import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';

class BackendApi {
  static const String _baseUrl = 'http://192.168.155.111:5000';
  final Dio _dio = Dio();

  Future<File?> uploadImage(File imageFile) async {
    try {
      String fileName = imageFile.path.split('/').last;
      FormData formData = FormData.fromMap({
        'image': await MultipartFile.fromFile(
          imageFile.path,
          filename: fileName,
        ),
      });

      Response response = await _dio.post(
        '$_baseUrl/upload',
        data: formData,
        options: Options(
          responseType: ResponseType.bytes,
          followRedirects: false,
          validateStatus: (status) => status! < 500,
        ),
      );

      if (response.statusCode == 200) {
        final tempDir = await getTemporaryDirectory();
        final file = File(
            '${tempDir.path}/processed_${DateTime.now().millisecondsSinceEpoch}.jpg');
        await file.writeAsBytes(response.data);
        return file;
      } else {
        return null;
      }
    } catch (e) {
      print('Error uploading image: $e');
      if (e is DioException) {}
      return null;
    }
  }

  Future<bool> checkFaceRecognitionHealth() async {
    try {
      Response response = await _dio.get('$_baseUrl/api/health');

      if (response.statusCode == 200 && response.data['status'] == 'ok') {
        return true;
      } else {
        print('Health check failed: ${response.data}');
        return false;
      }
    } catch (e) {
      print('Health check error: $e');
      return false;
    }
  }

  Future<Map<String, dynamic>?> recognizeFace(File imageFile) async {
    try {
      String fileName = imageFile.path.split('/').last;
      FormData formData = FormData.fromMap({
        'image': await MultipartFile.fromFile(
          imageFile.path,
          filename: fileName,
        ),
      });

      Response response = await _dio.post(
        '$_baseUrl/api/recognize',
        data: formData,
        options: Options(
          contentType: 'multipart/form-data',
          validateStatus: (status) => status! < 500,
        ),
      );

      if (response.statusCode == 200) {
        return response.data;
      } else {
        print(
            'Face recognition failed: ${response.statusCode} - ${response.data}');
        return null;
      }
    } catch (e) {
      print('Error recognizing face: $e');
      return null;
    }
  }

  // New function to get result files
  Future<File?> getResultFile(String filename) async {
    try {
      Response response = await _dio.get(
        '$_baseUrl/api/results/$filename',
        options: Options(
          responseType: ResponseType.bytes,
          validateStatus: (status) => status! < 500,
        ),
      );

      if (response.statusCode == 200) {
        final tempDir = await getTemporaryDirectory();
        final file = File('${tempDir.path}/$filename');
        await file.writeAsBytes(response.data);
        return file;
      } else {
        print('Failed to get result file: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('Error getting result file: $e');
      return null;
    }
  }

  // Helper function to download a base64 image from recognition result
  Future<File?> saveBase64Image(String base64String, String filename) async {
    try {
      final bytes = base64Decode(base64String);
      final tempDir = await getTemporaryDirectory();
      final file = File('${tempDir.path}/$filename');
      await file.writeAsBytes(bytes);
      return file;
    } catch (e) {
      print('Error saving base64 image: $e');
      return null;
    }
  }

  // Helper function that processes full recognition results
  // Returns a map with the processed results including local file paths
  Future<Map<String, dynamic>> processRecognitionResults(
      Map<String, dynamic> results) async {
    Map<String, dynamic> processedResults = Map.from(results);

    // Process base64 images if available
    if (results.containsKey('comparison_image_base64') &&
        results['comparison_image_base64'] != null) {
      final comparisonFile = await saveBase64Image(
          results['comparison_image_base64'],
          'comparison_${DateTime.now().millisecondsSinceEpoch}.jpg');

      if (comparisonFile != null) {
        processedResults['comparison_image_local_path'] = comparisonFile.path;
      }
    }

    // Download any other image files referenced by path
    for (String key in ['test_image', 'matched_image']) {
      if (results.containsKey(key) && results[key] != null) {
        final filename = results[key].split('/').last;
        final file = await getResultFile(filename);
        if (file != null) {
          processedResults['${key}_local_path'] = file.path;
        }
      }
    }

    return processedResults;
  }
}
