console.log("ml5 version:", ml5.version);

let video;
const knnClassifier = ml5.KNNClassifier();
let poseNet;
let poses = [];

let resultLabel = "";
let resultScore = 0.0;

let aScore = 0.0;
let bScore = 0.0;
let cScore = 0.0;
let dScore = 0.0;
let aCount = 0;
let bCount = 0;
let cCount = 0;
let dCount = 0;

let hi;
let questionMark;
let okay;
let nice;
function preload() {
  hi = loadImage(
    "images/hi.png"
  );
  questionMark = loadImage(
    "images/question.png"
  );
  okay = loadImage(
    "images/ok.png"
  );
  nice = loadImage(
    "images/nice.png"
  );
}

function setup() {
  createCanvas(640, 480);
  background(50);

  video = createCapture(VIDEO);
  video.hide();

  poseNet = ml5.poseNet(video, modelReady);
  poseNet.on("pose", function(results) {
    poses = results;
  });
}

function draw() {
  background(50);
  image(video, 0, 0);

  fill(0);

  // text(resultLabel + ": " + resultScore, 10, 20);

  text("Hi!: " + aCount, 10, 20);
  text("???: " + bCount, 10, 40);
  text("OK: " + cCount, 10, 60);
  text("Nice: " + dCount, 10, 80);

  if (poses.length > 0) {
    let currentPose = poses[0].pose;

    if (resultLabel == "A" && aScore > 95) {
      let x = currentPose.rightWrist.x;
      let y = currentPose.rightWrist.y;
      image(hi, x - 50, y - 200, 300, 240);
    }
    if (resultLabel == "B" && bScore > 95) {
      let x = currentPose.nose.x;
      let y = currentPose.nose.y;
      image(questionMark, x-50, y-50, 240, 300);
    }
    if (resultLabel == "C" && cScore > 95) {
      let x = currentPose.nose.x;
      let y = currentPose.nose.y;
      image(okay, x - 90, y-150, 150, 120);
    }
    if (resultLabel == "D" && dScore > 95) {
      let x = currentPose.rightShoulder.x;
      let y = currentPose.rightShoulder.y;
      image(nice, x-50, y-50, 300, 240);
    }
  }
}

function keyPressed() {
  //
  if (key === "a") {
    addExample("A");
  } else if (key === "b") {
    addExample("B");
  } else if (key === "c") {
    addExample("C");
  } else if (key === "d") {
    addExample("D");
  }
  // key press with shift
  else if (key === "A") {
    clearLabel("A");
  } else if (key === "B") {
    clearLabel("B");
  } else if (key === "C") {
    clearLabel("C");
  } else if (key === "D") {
    clearLabel("D");
  }
  //
  else if (key == " ") {
    classify();
  } else if (key == "S") {
    saveMyKNN();
    console.log("Saved");
  } else if (key == "L") {
    loadMyKNN();
    console.log("Loaded");
  }
}

function modelReady() {
  console.log("Model Loaded");
}

// Predict the current frame.
function classify() {
  // Get the total number of labels from knnClassifier
  const numLabels = knnClassifier.getNumLabels();
  if (numLabels <= 0) {
    console.error("There is no examples in any label");
    return;
  }
  // Get the features of the input video
  const poseArray = poses[0].pose.keypoints.map(p => [
    p.score,
    p.position.x,
    p.position.y
  ]);
  knnClassifier.classify(poseArray, gotResults);
}

// Show the results
function gotResults(err, result) {
  // Display any error
  if (err) {
    console.error(err);
  }

  if (result.confidencesByLabel) {
    const confidences = result.confidencesByLabel;
    // result.label is the label that has the highest confidence
    if (result.label) {
      resultLabel = result.label;
      resultScore = confidences[result.label] * 100;
    }
    aScore = confidences["A"] * 100;
    bScore = confidences["B"] * 100;
    cScore = confidences["C"] * 100;
    dScore = confidences["D"] * 100;
  }

  classify();
}

function addExample(label) {
  const poseArray = poses[0].pose.keypoints.map(p => [
    p.score,
    p.position.x,
    p.position.y
  ]);
  // Add an example with a label to the classifier
  console.log(poseArray);
  knnClassifier.addExample(poseArray, label);
  updateCounts();
}

// Clear the examples in one label
function clearLabel(label) {
  knnClassifier.clearLabel(label);
  updateCounts();
}

// Clear all the examples in all labels
function clearAllLabels() {
  knnClassifier.clearAllLabels();
  updateCounts();
}

// Save dataset as myKNNDataset.json
function saveMyKNN() {
  knnClassifier.save("myKNNDataset");
}

// Load dataset to the classifier
function loadMyKNN() {
  knnClassifier.load("./myKNNDataset.json", updateCounts);
}

function updateCounts() {
  const counts = knnClassifier.getCountByLabel();
  aCount = counts["A"];
  bCount = counts["B"];
  cCount = counts["C"];
  dCount = counts["D"];
}

