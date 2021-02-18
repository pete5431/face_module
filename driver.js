const youauth = require('./node_modules/youauth');

async function main(){

  await youauth.loadModels();

  const newRecognizer = new youauth.FaceRecognizer;

  const refImages = ['images/biden.jpg'];
  const labels = ['biden'];

  const labeledFaceDescriptors = await newRecognizer.labelDescriptors(labels, refImages);

  console.log(labeledFaceDescriptors);

  const targetImage = await newRecognizer.loadImage('images/test_image3.jpg');

  console.log(targetImage);

  let detectedResults = await newRecognizer.detect(targetImage);

  console.log(detectedResults);

  let matches = newRecognizer.getMatches(detectedResults, labeledFaceDescriptors);

  let matchedLabels = newRecognizer.getMatchedLabels(matches);

  console.log(matchedLabels);

}

main();
