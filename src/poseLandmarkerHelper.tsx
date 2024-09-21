/* eslint-disable @typescript-eslint/no-explicit-any */
import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils,
  Landmark,
} from "@mediapipe/tasks-vision";

type Point = {
  x: number;
  y: number;
};

export const initializePoseLandmarker = async (
  videoRef: React.RefObject<HTMLVideoElement>,
  canvasRef: React.RefObject<HTMLCanvasElement>,
  setLandmarksData: React.Dispatch<React.SetStateAction<string[][]>>,
  landmarkNames: string[],
  setIsHugging: React.Dispatch<React.SetStateAction<boolean>>,
  setIsSideHugging: React.Dispatch<React.SetStateAction<boolean>>,
  setIsPersonDetected: React.Dispatch<React.SetStateAction<boolean>>,
  PARAMS: any
): Promise<PoseLandmarker> => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1, // Detect single person
    minPoseDetectionConfidence: PARAMS.minPoseDetectionConfidence,
    minTrackingConfidence: PARAMS.minTrackingConfidence,
  });

  // Start the pose prediction using the webcam
  await predictWebcam(
    poseLandmarker,
    videoRef,
    canvasRef,
    setLandmarksData,
    landmarkNames,
    setIsHugging,
    setIsSideHugging,
    setIsPersonDetected,
    PARAMS
  );

  return poseLandmarker;
};

const calculateDistance = (point1: Point, point2: Point): number => {
  const dx = point1.x - point2.x;
  const dy = point1.y - point2.y;
  return Math.sqrt(dx * dx + dy * dy);
};

const calculateAngle = (a: Point, b: Point, c: Point): number => {
  // Calculates the angle at point b, between points a, b, and c
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dotProduct = ab.x * cb.x + ab.y * cb.y;
  const magnitudeAB = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const magnitudeCB = Math.sqrt(cb.x * cb.x + cb.y * cb.y);

  const angleRadians = Math.acos(dotProduct / (magnitudeAB * magnitudeCB));
  return angleRadians * (180 / Math.PI); // Convert to degrees
};

// Function to determine if the pose is valid (not on the blacklist)
const isPoseValid = (landmarks: Landmark[]): boolean => {
  const head = landmarks[0];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];

  // Head should be above hips
  const isUpright = head.y < (leftHip.y + rightHip.y) / 2;

  // Key landmarks should be visible
  const keyLandmarksVisible = landmarks
    .slice(0, 5) // Adjust as needed
    .every((landmark) => landmark.visibility && landmark.visibility > 0.5);

  return isUpright && keyLandmarksVisible;
};

const isHugging = (landmarks: Landmark[], PARAMS: any): boolean => {
  // Swap left and right landmarks due to mirroring
  const leftWrist = landmarks[16]; // rightWrist
  const rightWrist = landmarks[15]; // leftWrist
  const leftElbow = landmarks[14]; // rightElbow
  const rightElbow = landmarks[13]; // leftElbow
  const leftShoulder = landmarks[12]; // rightShoulder
  const rightShoulder = landmarks[11]; // leftShoulder
  const leftHip = landmarks[24]; // rightHip
  const rightHip = landmarks[23]; // leftHip

  const leftWristToRightShoulder = calculateDistance(leftWrist, rightShoulder);
  const rightWristToLeftShoulder = calculateDistance(rightWrist, leftShoulder);

  const huggingThreshold = PARAMS.frontHugValue;

  // Existing condition: Wrists are below shoulders
  const wristsBelowShoulders =
    leftWrist.y - leftShoulder.y > PARAMS.wristsBelowShouldersThreshold &&
    rightWrist.y - rightShoulder.y > PARAMS.wristsBelowShouldersThreshold;

  // New condition: Wrists are above elbows
  const leftWristAboveElbow =
    leftElbow.y - leftWrist.y > PARAMS.wristAboveElbowThreshold;
  const rightWristAboveElbow =
    rightElbow.y - rightWrist.y > PARAMS.wristAboveElbowThreshold;
  const wristsAboveElbows = leftWristAboveElbow && rightWristAboveElbow;

  // New condition: Shoulders are above hips
  const leftShoulderAboveHip =
    leftHip.y - leftShoulder.y > PARAMS.shoulderAboveHipThreshold;
  const rightShoulderAboveHip =
    rightHip.y - rightShoulder.y > PARAMS.shoulderAboveHipThreshold;
  const shouldersAboveHips = leftShoulderAboveHip && rightShoulderAboveHip;

  // Adjusted condition: Wrists are not too close to hips
  const leftWristAboveHip =
    leftHip.y - leftWrist.y > PARAMS.wristAboveHipThreshold;
  const rightWristAboveHip =
    rightHip.y - rightWrist.y > PARAMS.wristAboveHipThreshold;
  const wristsAboveHips = leftWristAboveHip && rightWristAboveHip;

  return (
    leftWristToRightShoulder < huggingThreshold &&
    rightWristToLeftShoulder < huggingThreshold &&
    wristsBelowShoulders &&
    wristsAboveElbows &&
    shouldersAboveHips &&
    wristsAboveHips
  );
};

// Adjusted isSideHugging function
const isSideHugging = (landmarks: Landmark[], PARAMS: any): boolean => {
  // Swap left and right landmarks due to mirroring
  const leftShoulder = landmarks[12]; // rightShoulder
  const rightShoulder = landmarks[11]; // leftShoulder
  const leftElbow = landmarks[14]; // rightElbow
  const rightElbow = landmarks[13]; // leftElbow
  const leftWrist = landmarks[16]; // rightWrist
  const rightWrist = landmarks[15]; // leftWrist
  const leftHip = landmarks[24]; // rightHip
  const rightHip = landmarks[23]; // leftHip

  const shoulderDistance = calculateDistance(leftShoulder, rightShoulder);

  const leftElbowToLeftHip = calculateDistance(leftElbow, leftHip);
  const rightElbowToRightHip = calculateDistance(rightElbow, rightHip);

  const shoulderThreshold = PARAMS.sideHugShoulderThreshold; // Adjustable
  const elbowHipThreshold = PARAMS.sideHugElbowHipThreshold; // Adjustable

  const isShouldersClose = shoulderDistance < shoulderThreshold;
  const isArmsCloseToTorso =
    leftElbowToLeftHip < elbowHipThreshold &&
    rightElbowToRightHip < elbowHipThreshold;

  // Calculate elbow angles
  const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
  const rightElbowAngle = calculateAngle(
    rightShoulder,
    rightElbow,
    rightWrist
  );

  const elbowBentThreshold = PARAMS.elbowBentThreshold; // New parameter, e.g., 150 degrees

  const isLeftElbowBent = leftElbowAngle < elbowBentThreshold;
  const isRightElbowBent = rightElbowAngle < elbowBentThreshold;

  const areElbowsBent = isLeftElbowBent && isRightElbowBent;

  return isShouldersClose && isArmsCloseToTorso && areElbowsBent;
};

const predictWebcam = async (
  poseLandmarker: PoseLandmarker,
  videoRef: React.RefObject<HTMLVideoElement>,
  canvasRef: React.RefObject<HTMLCanvasElement>,
  setLandmarksData: React.Dispatch<React.SetStateAction<string[][]>>,
  landmarkNames: string[],
  setIsHugging: React.Dispatch<React.SetStateAction<boolean>>,
  setIsSideHugging: React.Dispatch<React.SetStateAction<boolean>>,
  setIsPersonDetected: React.Dispatch<React.SetStateAction<boolean>>,
  PARAMS: any
) => {
  const canvasCtx = canvasRef.current?.getContext("2d");

  if (!canvasCtx) {
    console.error("Canvas context is not available.");
    return;
  }

  const drawingUtils = new DrawingUtils(canvasCtx);

  // Initialize smoothing variables
  const landmarkBuffer: Landmark[][] = [];
  const bufferSize = 5; // Number of frames to buffer

  const render = async () => {
    const video = videoRef.current!;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.warn("Invalid video dimensions, skipping frame.");
      requestAnimationFrame(render);
      return;
    }

    const startTimeMs = performance.now();
    const vWidth = video.videoWidth;
    const vHeight = video.videoHeight;

    // Ensure canvas matches video size
    canvasRef.current!.width = vWidth;
    canvasRef.current!.height = vHeight;

    try {
      const results = await poseLandmarker.detectForVideo(video, startTimeMs);

      canvasCtx.clearRect(0, 0, vWidth, vHeight);

      const newLandmarksData: string[][] = [[], []];

      if (results.landmarks && results.landmarks.length > 0) {
        setIsPersonDetected(true);
      } else {
        setIsPersonDetected(false);
        setIsHugging(false);
        setIsSideHugging(false);
      }

      results.landmarks?.forEach(
        (landmarks: Landmark[], personIndex: number) => {
          // Add current landmarks to buffer
          if (landmarkBuffer.length >= bufferSize) {
            landmarkBuffer.shift(); // Remove the oldest frame
          }
          landmarkBuffer.push(landmarks);

          // Compute the averaged landmarks
          const smoothedLandmarks = landmarks.map((_, idx) => {
            let sumX = 0,
              sumY = 0,
              sumZ = 0;
            landmarkBuffer.forEach((bufferedLandmarks) => {
              sumX += bufferedLandmarks[idx].x;
              sumY += bufferedLandmarks[idx].y;
              sumZ += bufferedLandmarks[idx].z;
            });
            const len = landmarkBuffer.length;
            return {
              x: sumX / len,
              y: sumY / len,
              z: sumZ / len,
              visibility: landmarks[idx].visibility,
            };
          });

          smoothedLandmarks.forEach(
            (landmark: Landmark, index: number) => {
              const x = Math.round(landmark.x * vWidth);
              const y = Math.round(landmark.y * vHeight);
              const z = Math.round(landmark.z * 100);

              const landmarkName = landmarkNames[index] || `Landmark ${index}`;
              newLandmarksData[personIndex].push(
                `${landmarkName}: x=${x}, y=${y}, z=${z}`
              );
            }
          );

          // Check if the pose is valid (not in blacklist poses)
          const isValidPose = isPoseValid(smoothedLandmarks);

          let isHugDetected = false;
          let isSideHugDetected = false;

          if (isValidPose) {
            isHugDetected = isHugging(smoothedLandmarks, PARAMS);
            isSideHugDetected = isSideHugging(smoothedLandmarks, PARAMS);
          } else {
            // Pose is invalid, set both detections to false
            isHugDetected = false;
            isSideHugDetected = false;
          }

          setIsHugging(isHugDetected);
          setIsSideHugging(isSideHugDetected);

          // Set color based on conditions
          let color: string;

          if (isHugDetected) {
            color = "green";
          } else if (isSideHugDetected) {
            color = "orange";
          } else if (isValidPose) {
            color = "blue";
          } else {
            color = "red";
          }

          if (PARAMS.Show_Pose_Detector) {
            drawingUtils.drawLandmarks(smoothedLandmarks, {
              radius: (data) =>
                DrawingUtils.lerp(data.from?.z || 0, -0.15, 0.1, 5, 1),
              color: color,
            });

            drawingUtils.drawConnectors(
              smoothedLandmarks,
              PoseLandmarker.POSE_CONNECTIONS,
              {
                color: color,
              }
            );

            // Visual debugging information
            canvasCtx.font = "16px Arial";
            canvasCtx.fillStyle = "white";
            canvasCtx.fillText(
              `Is Hugging: ${isHugDetected ? "Yes" : "No"}`,
              10,
              30
            );
            canvasCtx.fillText(
              `Is Side Hugging: ${isSideHugDetected ? "Yes" : "No"}`,
              10,
              50
            );
          }
        }
      );

      setLandmarksData(newLandmarksData);
    } catch (error) {
      console.error("Error during pose detection:", error);
    }

    requestAnimationFrame(render);
  };

  render();
};

export {
  predictWebcam,
  isHugging,
  isSideHugging,
  isPoseValid,
  calculateDistance,
  calculateAngle,
};
