import {
  FilesetResolver,

  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { RefObject, Dispatch, SetStateAction } from "react";

export const initializeVideoStream = async (
  videoRef: React.RefObject<HTMLVideoElement>
): Promise<MediaStream | null> => {
  try {
    const vWidth = window.innerWidth;
    const vHeight = window.innerHeight;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: vWidth, height: vHeight },
    });

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await new Promise((resolve) => {
        videoRef.current!.onloadeddata = () => resolve(true);
      });
      videoRef.current.play();
    }

    return stream;
  } catch (error) {
    console.error("Error initializing video stream:", error);
    return null;
  }
};

// Remove this or call it appropriately in the code
// function renderLoop(
//   objectDetector: ObjectDetector,
//   videoRef: RefObject<HTMLVideoElement>,
//   canvasRef: RefObject<HTMLCanvasElement>,
//   setDetectedObjects: Dispatch<SetStateAction<string[]>>
// ) {
//   const canvas = canvasRef.current;
//   const video = videoRef.current;
//   if (!canvas || !video) return;

//   const ctx = canvas.getContext("2d");
//   if (!ctx) return;

//   const detectObjects = async () => {
//     const videoFrame = video as HTMLVideoElement;

//     // Run object detection
//     const results = await objectDetector.detectForVideo(videoFrame, Date.now());

//     // Clear the canvas
//     ctx.clearRect(0, 0, canvas.width, canvas.height);

//     // Draw the detected objects on the canvas
//     results.detections.forEach((detection) => {
//       const bbox = detection.boundingBox;
//       if (bbox) {
//         ctx.beginPath();
//         ctx.rect(bbox.originX, bbox.originY, bbox.width, bbox.height);
//         ctx.lineWidth = 2;
//         ctx.strokeStyle = "green";
//         ctx.stroke();
//         ctx.fillStyle = "white";
//         const label = detection.categories[0].categoryName;
//         ctx.fillText(label, bbox.originX, bbox.originY - 10);
//       }
//     });

//     // Update detected objects
//     const detectedLabels = results.detections.map(
//       (detection) => detection.categories[0].categoryName
//     );
//     setDetectedObjects(detectedLabels);

//     // Call detect again on the next animation frame
//     requestAnimationFrame(detectObjects);
//   };

//   // Start the detection loop
//   detectObjects();
// }

function predictWebcam(
  poseLandmarker: PoseLandmarker,
  videoRef: RefObject<HTMLVideoElement>,
  canvasRef: RefObject<HTMLCanvasElement>,
  setLandmarksData: Dispatch<SetStateAction<string[][]>>,
  landmarkNames: string[]
) {
  const canvas = canvasRef.current;
  const video = videoRef.current;
  if (!canvas || !video) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let lastDetectionTime = 0;
  const detectionInterval = 200; // Throttle detection to every 200ms

  const detectPoses = async () => {
    const currentTime = performance.now();
    if (currentTime - lastDetectionTime > detectionInterval) {
      const videoFrame = video as HTMLVideoElement;

      // Run pose detection
      const results = await poseLandmarker.detectForVideo(videoFrame, Date.now());

      // Clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw the detected poses on the canvas
      results.landmarks.forEach((landmarks, poseIndex) => {
        ctx.fillStyle = "red";
        landmarks.forEach((landmark) => {
          ctx.beginPath();
          ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
          ctx.fill();
        });

        // Update landmarks data for external use
        const landmarkData = landmarks.map((landmark, index) => [
          landmarkNames[index] || `Landmark ${index + 1}`,
          `X: ${landmark.x.toFixed(3)}`,
          `Y: ${landmark.y.toFixed(3)}`,
        ]);
        
        setLandmarksData((prevState) => {
          const newState = [...prevState];
          newState[poseIndex] = landmarkData.flat();  // Use .flat() to flatten the 2D array into a 1D array
          return newState;
        });
      });

      lastDetectionTime = currentTime;
    }

    // Call detect again on the next animation frame
    requestAnimationFrame(detectPoses);
  };

  // Start the pose detection loop
  detectPoses();
}

export const initializePoseLandmarker = async (
  videoRef: React.RefObject<HTMLVideoElement>,
  canvasRef: React.RefObject<HTMLCanvasElement>,
  setLandmarksData: React.Dispatch<React.SetStateAction<string[][]>>,
  landmarkNames: string[],
  minPoseDetectionConfidence: number,  
  minTrackingConfidence: number        
) => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  // If an existing PoseLandmarker instance exists, close it
  if (window.currentPoseLandmarker) {
    window.currentPoseLandmarker.close();
  }

  const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 2,
    minPoseDetectionConfidence: minPoseDetectionConfidence, 
    minTrackingConfidence: minTrackingConfidence,            
  });

  window.currentPoseLandmarker = poseLandmarker;  // Store the current instance globally

  const video = videoRef.current;
  const canvas = canvasRef.current;

  if (video && canvas) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    predictWebcam(
      poseLandmarker,
      videoRef,
      canvasRef,
      setLandmarksData,
      landmarkNames
    );
  }

  return poseLandmarker;
};
