const youauth = require('./youauth');

async function main(){

  await youauth.loadModels();

  const newRecognizer = new youauth.FaceRecognizer;

  const refImages = ['cap.jpg'];
  const labels = ['cap'];

  const labeledFaceDescriptors = await newRecognizer.labelDescriptors(labels, refImages);

  console.log(labeledFaceDescriptors);

  const targetImage = await newRecognizer.loadImage('test_image7.jpg');

  let detectedResults = await newRecognizer.detect(targetImage);

  let matches = newRecognizer.getMatches(detectedResults, labeledFaceDescriptors);

  let matchedLabels = newRecognizer.getMatchedLabels(matches);

  const drawnTensor = await newRecognizer.drawFaceDetections(matches, detectedResults, targetImage);

  console.log(drawnTensor);

  newRecognizer.saveImageJPG(drawnTensor, "testTensorOutput2.jpg");

  console.log(matches);

  console.log(matchedLabels);
}

main();
