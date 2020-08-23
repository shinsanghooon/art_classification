import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as IMG;
import 'package:image_picker/image_picker.dart';
import 'package:flutter/services.dart';
import 'package:tflite/tflite.dart';
import 'package:path_provider/path_provider.dart';
//import 'package:firebase_ml_custom/firebase_ml_custom.dart';


class cameraPage extends StatefulWidget {
  @override
  _cameraPageState createState() => _cameraPageState();
}

class _cameraPageState extends State<cameraPage> {

  bool _isLoading;
  File _image;
  List _outputs;

  final ImagePicker picker = ImagePicker();
  String answer;

  @override
  void initState() {
    super.initState();
    _isLoading = true;
    loadTensorFlowLiteModel().then((value){
      setState(() {
        _isLoading = false;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title : 'Art',
      home : Scaffold(
          body: Center(
              child:
              SingleChildScrollView(
                  child: Center(
                      child: Column(
                          children: <Widget>[
                            Text("궁금한 작품을 찾아보세요.",
                                style: TextStyle(fontSize: 24.0)),
                            Padding(padding: EdgeInsets.all(8.0)),
                            Text("사진을 찍거나 앨범에 있는 사진을 선택하세요."),
                            Padding(padding: EdgeInsets.all(10.0)),
                            Row(
                                mainAxisAlignment: MainAxisAlignment
                                    .spaceAround,
                                children: <Widget>[
                                  RaisedButton(
                                      child: Text("작품 찍기"),
                                      color: Colors.indigoAccent,
                                      textColor: Colors.white,
                                      onPressed: () {
                                        _getImage(ImageSource.camera);
                                      }),
                                  RaisedButton(
                                      child: Text("작품 불러오기"),
                                      color: Colors.indigoAccent,
                                      textColor: Colors.white,
                                      onPressed: () {
                                        _getImage(ImageSource.gallery);
                                      }),
                                ]),
                            Padding(padding: EdgeInsets.all(25.0)),
                            _image == null ? Container() : Image.file(_image),
                            SizedBox(
                              height: 20,
                            ),
                            Padding(padding: EdgeInsets.all(10.0)),
                            _outputs != null
                                ? Text(
                              "${_outputs[0]["label"]}",
                              style: TextStyle(
                                color: Colors.black,
                                fontSize: 20.0,
                              ),
                            )
                                : Container()
                          ]
                      )))
          )
      ),
    );
  }

  Future _getImage(ImageSource imageSource) async {
    var image = await picker.getImage(source: imageSource);
    setState(() {
      _isLoading = true;
      _image = File(image.path);
    });
    classifyImage(image.path);
  }

  classifyImage(img) async {

//    Resize Image
//    _modelImage = IMG.decodeImage(File(img).readAsBytesSync());
//    _modelImage = IMG.copyResize(_modelImage, width: 64, height:64);
//
//    Directory tempDir = await getTemporaryDirectory();
//    String tempPath = tempDir.path;
//    _modelFile = new File(tempPath + '/temp.png')..writeAsBytesSync(IMG.encodePng(_modelImage));
//    print(_modelFile.path);

    var outputs = await Tflite.runModelOnImage(
        path: img,//_modelFile.path,   // required
        imageMean: 0.0,   // defaults to 117.0
        imageStd: 255.0,  // defaults to 1.0
        numResults: 3,    // defaults to 5
        threshold: 0.2,   // defaults to 0.1
    );
    print('Answer Candidates: $outputs');

    if (outputs.length == 0) {
      answer = 'No answer';
    } else {
      answer = outputs[0]['label'];
    }

    setState(() {
      _isLoading = false;
      _outputs = outputs;
    });

  }

  loadTensorFlowLiteModel() async {
    await Tflite.loadModel(model: "assets/tf_1783_32_normalize.tflite",
        labels: "assets/labels.txt",
        numThreads: 1, // defaults to 1
        isAsset: true, // defaults to true, set to false to load resources outside assets
        useGpuDelegate: false // defaults to false, set to true to use GPU delegate
    );
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

}
