import 'dart:io';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';

class BackendApi {
  static const String _baseUrl =
      'http://192.168.1.19:5000'; // Replace with your server IP
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
        print('Server responded with status code: ${response.statusCode}');
        print('Response data: ${response.data}');
        return null;
      }
    } catch (e) {
      print('Error uploading image: $e');
      if (e is DioException) {
        print('Dio error details:');
        print('  Response: ${e.response}');
        print('  Message: ${e.message}');
        print('  Type: ${e.type}');
      }
      return null;
    }
  }
}
