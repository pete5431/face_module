const faceAPI = require('face-api.js');
// Not required, but makes the code faster by connecting to a tensoflow node backend.
const tf = require('@tensorflow/tfjs-node');
const { createCanvas, createImageData, Image } = require('canvas');
const fs = require('fs');
const path = require('path');

// Path to the face detection, face recognition, etc. models.
const modelPath = path.resolve(__dirname, './models');

// Loads the models for face-api. Must be called first before using api.
async function loadModels(){
  await faceAPI.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceAPI.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceAPI.nets.faceRecognitionNet.loadFromDisk(modelPath);
  console.log('Models loaded.');
  return true;
}

// FaceRecognizer Object constructor.
function FaceRecognizer(){
  // Minimum confidence for valid face. Higher means less chance of bad face detections (less accurate face descriptors).
  this.minConfidence = 0.8;
  // The euclidean distance threshold. Lower means faceMatcher will look for more accurate match.
  this.distanceThreshold = 0.6;
  // Use SSD MobileNet Face Detection for higher accuracy. Uses minConfidence to set face detection threshold.
  this.options = new faceAPI.SsdMobilenetv1Options({minConfidence: this.minConfidence });
}

// Gather descriptive features of reference images.
FaceRecognizer.prototype.labelDescriptors = function labelDescriptors(labels, refImages){
  /*
   * labels is an array of names.
   * refImages is an array of dataURLs or Image Objects.
   * Will detect single face of highest confidence in each image. Throws error if no face is found.
   * The descriptor (Float32Array) can be thought of as extracted facial embeddings.
   * Example of returned result:
   * [
   *  LabeledFaceDescriptors {
   *    _label: 'person1',
   *    _descriptors: [ [Float32Array] ]
   *  },
   *  LabeledFaceDescriptors {
   *    _label: 'person2',
   *    _descriptors: [ [Float32Array] ]
   *  }
   * ]
  */
  // Make sure labels and regImages are same length.
  if(labels.length !== refImages.length){
    console.error('Labels and images not aligned!');
    return;
  }
  // Return array of LabeledFaceDescriptors objects. Each has a label and a Float32Array (descriptors).
  return Promise.all(
    labels.map(async(label, i) =>{
      // Create Image object from dataURL or the filePath.
      const img = await this.loadImage(refImages[i]);
      // Use face-api.js to detect a single face. Returns face of highest confidence even if multiple faces detected.
      const result = await faceAPI.detectSingleFace(img, this.options).withFaceLandmarks().withFaceDescriptor();
      // If no face detected, or confidence is too low, return undefined.
      if(result === undefined){
        console.error('No faces detected.');
        return;
      }
      // Fetch the descriptor value (face embeddings) of result.
      const faceDescriptor = [result.descriptor];
      // Create new LabeledFaceDescriptors object using the associated label and face descriptor.
      return new faceAPI.LabeledFaceDescriptors(label, faceDescriptor);
    })
  );
}

// Load the labeled descriptors from the JSON.stringified LabeledFaceDescriptors object.
FaceRecognizer.prototype.loadDescriptors = function loadDescriptors(jsonString){
  /*
   * Contents example (what the array contents should look like after parsing):
   * [
   *  { label: 'person1', descriptors: [ [Array] ] },
   *  { label: 'person2', descriptors: [ [Array] ] }
   * ]
  */
  // Parse the json.
  var contents = JSON.parse(jsonString);
  // Initialize new labeledFaceDescriptors array.
  labeledFaceDescriptors = [];
  // For each (label, descriptors).
  contents.forEach((content, i) => {
    // Create new [Float32Array].
    let values = new Float32Array(content.descriptors[0]);
    let arr = [values];
    // Create new LabeledFaceDescriptors using the label and descriptor.
    labeledFaceDescriptors[i] = new faceAPI.LabeledFaceDescriptors(content.label, arr);
  })
  return labeledFaceDescriptors;
}

// Returns array of labels that were found in matches.
FaceRecognizer.prototype.getMatchedLabels = function getMatchedLabels(matches){
  /*
   * matches is an array of FaceMatch objects from face-api.js.
   * Each FaceMatch contains a string: label, and a number: distance.
   * Example:
   * [
   *  FaceMatch { _label: 'person1', _distance: 0.35212970923781933 },
   *  FaceMatch { _label: 'person2', _distance: 0.4249780266473695 }
   * ]
  */
  // Create an array of labels that were found (aka not unknown).
  matchedLabels = [];
  // For each FaceMatch object in matches.
  matches.forEach((match, i) => {
    // Get the label of each match.
    label = match._label;
    // By default unknown faces are labeled 'unknown'.
    if(label !== 'unknown'){
      // Add the found label to the array.
      matchedLabels.push(label);
    }
  });
  return matchedLabels;
}

// Detects faces in image using face-api.js.
FaceRecognizer.prototype.detect = async function detect(img){
  // Detect all faces that aren't below the minConfidence threshold.
  const detectionResults = await faceAPI.detectAllFaces(img, this.options).withFaceLandmarks().withFaceDescriptors();
  return detectionResults;
}

// Get the matches from face detection results using face-api.js.
FaceRecognizer.prototype.getMatches = function getMatches(detectionResults, labeledFaceDescriptors){
  // Create a faceMatcher using the labeled descriptors.
  const faceMatcher = new faceAPI.FaceMatcher(labeledFaceDescriptors, this.distanceThreshold);
  // Map the descriptors in the results with the descriptor of best match.
  const matches = detectionResults.map(fd => faceMatcher.findBestMatch(fd.descriptor));
  return matches;
}

// Saved the labeled face descriptors as a json file for later use.
FaceRecognizer.prototype.saveDescriptors = function saveDescriptors(labeledFaceDescriptors, filePath){
  // stringify the array.
  const jsonString = JSON.stringify(labeledFaceDescriptors);
  // Write file to the path.
  fs.writeFileSync(filePath, jsonString);
}

// Save images to folder. Right now has a default path.
FaceRecognizer.prototype.saveImageFile = function saveImageFile(fileName, imageData){
  // Remove the image data header.
  var data = imageData.replace(/^data:image\/\w+;base64,/, "");
  // Convert to buffer (sequence of bytes) from base64 (what the image data is encrypted as).
  var buf = Buffer.from(data, 'base64');
  // Save file to images folder.
  fs.writeFileSync(path.resolve('./images', fileName), buf);
}

// Creates tensor out of image data.
FaceRecognizer.prototype.loadImage = async function loadImage(fileName){

  const filePath = "images/" + fileName;

  // Read file as buffer.
  const buf = fs.readFileSync(filePath);
  /*
    Create tensor out of buffer using tfjs-node decodeImage function.
    Arguments: Uint8Array (decoded image), color channel (default 0), data type (only int32 supported)
  */
  tensor = tf.node.decodeImage(buf, 3, 'int32');
  return tensor;
}

FaceRecognizer.prototype.loadImageData = async function loadImageData (data) {

  var data = data.replace(/^data:image\/\w+;base64,/, "");

  var buf = Buffer.from(data, 'base64');

  tensor = tf.node.decodeImage(buf, 3, 'int32');
  return tensor;
}

// Saves tensor as JPG image file.
FaceRecognizer.prototype.saveImageJPG = function saveImageJPG(tensor, fileName) {

  const filePath = "images/" + fileName;

  tf.node.encodeJpeg(tensor).then ( (fileData) => {
    fs.writeFileSync(filePath, fileData);
  });

}

/* Draw matches on image. Takes matches from getMatches() and results from detect().
   The outputCanvas is from createCanvas().
   Handy for seeing the outcome of the detection results. */
FaceRecognizer.prototype.drawFaceDetections = async function drawFaceDetections(matches, results, tensor){

  const data = await tf.node.encodeJpeg(tensor);

  var buf = Buffer.from(data, 'base64');

  const newCanvas = createCanvas(tensor.shape[1], tensor.shape[0]);

  const image = new Image;

  image.onload = function () {
    newCanvas.getContext("2d").drawImage(image, 0, 0);
  };

  image.src = buf;

  // For each match object in matches.
  matches.forEach((match, i) => {
    // Grab associated bounding box from the face detection in results.
    const box = results[i].detection.box;
    // Get label associated with match. Unknown faces with no label are 'unknown' by default.
    const label = match.toString();

    const ctx = newCanvas.getContext("2d");

    ctx.strokeStyle = "blue";
    ctx.font = '10px sans';
    ctx.strokeText(label, box._x + 10, box._y - 5);
    ctx.strokeRect(box._x, box._y, box._width, box._height);
  });

  const dataURL = newCanvas.toDataURL();

  return this.loadImageData(dataURL);
}

module.exports = {
  FaceRecognizer: FaceRecognizer,
  loadModels: loadModels
}
