import { useEffect, useRef, useState } from "react";
import { Pane } from "tweakpane";
import { initializePoseLandmarker } from "./poseLandmarkerHelper";
import { initializeVideoStream } from "./videoStreamHelper";
import { PoseLandmarker } from "@mediapipe/tasks-vision";

// Extend window object to include currentPoseLandmarker
declare global {
  interface Window {
    currentPoseLandmarker: PoseLandmarker | null;
  }
}

// Updated debounce function
const debounce = (
  func: (...args: unknown[]) => void,
  delay: number
): ((...args: unknown[]) => void) => {
  let timeoutId: number;
  return (...args: unknown[]) => {
    clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => {
      func(...args);
    }, delay);
  };
};

function ObjectAndPoseDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const poseCanvasRef = useRef<HTMLCanvasElement>(null);

  const [, setLandmarksData] = useState<string[][]>([[], []]);
  const [isHugging, setIsHugging] = useState<boolean>(false);
  const [isSideHugging, setIsSideHugging] = useState<boolean>(false);
  const [isPersonDetected, setIsPersonDetected] = useState<boolean>(false);
  const [showIntroText, setShowIntroText] = useState<boolean>(false);

  const paneRef = useRef<Pane | null>(null);
  const PARAMS = useRef({
    Show_Pose_Detector: true,
    VolumeIncreaseHD: 100,
    minPoseDetectionConfidence: 0.6, // Increased from 0.4
    minTrackingConfidence: 0.8, // Increased from 0.6
    frontHugValue: 0.3,
    wristsBelowShouldersThreshold: 0.0,
    wristAboveElbowThreshold: -0.05,
    shoulderAboveHipThreshold: 0.2,
    wristAboveHipThreshold: 0.1,
    sideHugShoulderThreshold: 0.1,
    sideHugElbowHipThreshold: 0.1,
    elbowBentThreshold: 120,
    Volume: 0,
    fadeSpeed: 0.3,
    showVideo: false,
    showCanvas: false,
  });

  const [showVideo, setShowVideo] = useState(PARAMS.current.showVideo);
  const [showCanvas, setShowCanvas] = useState(PARAMS.current.showCanvas);
  const [videoOpacity, setVideoOpacity] = useState(0);
  const [isStarted, setIsStarted] = useState(false);

  const audioFilePath = "/music/NSA.mp3";

  // PoseLandmarker initialization logic
  const startDetection = async () => {
    const stream = await initializeVideoStream(videoRef);

    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      if (window.currentPoseLandmarker) {
        return;
      }

      const poseLandmarker = await initializePoseLandmarker(
        videoRef,
        poseCanvasRef,
        setLandmarksData,
        [],
        setIsHugging,
        setIsSideHugging,
        setIsPersonDetected,
        PARAMS.current
      );

      window.currentPoseLandmarker = poseLandmarker;
    }
  };

  // Throttled function to update only the detection parameters, not the whole landmarker
  const updatePoseLandmarkerParams = debounce(() => {
    if (window.currentPoseLandmarker) {
      window.currentPoseLandmarker.setOptions({
        minPoseDetectionConfidence: PARAMS.current.minPoseDetectionConfidence,
        minTrackingConfidence: PARAMS.current.minTrackingConfidence,
      });
    }
  }, 300);

  // Create the Tweakpane controls (hidden from the user)
  useEffect(() => {
    if (!paneRef.current) {
      const pane = new Pane({ title: "Controls" });
      paneRef.current = pane;

      // Add your tweakpane controls
      const f1 = pane.addFolder({ title: "Music" });
      const f2 = pane.addFolder({ title: "Hug Detector" });
      const f3 = pane.addFolder({ title: "Display Options" });

      f1.addBinding(PARAMS.current, "Volume", {
        label: "Volume:",
        readonly: true,
      });
      f1.addBinding(PARAMS.current, "Volume", {
        label: "Volume:",
        readonly: true,
        view: "graph",
        min: 0,
        max: 100,
      });
      f1.addBinding(PARAMS.current, "fadeSpeed", {
        label: "Fade Speed:",
        min: 0.01,
        max: 1.0,
        step: 0.01,
      });

      f2
        .addBinding(PARAMS.current, "minPoseDetectionConfidence", {
          min: 0.1,
          max: 1.0,
          step: 0.1,
          label: "Overall Tolerance:",
        })
        .on("change", () => {
          updatePoseLandmarkerParams();
        });

      f2
        .addBinding(PARAMS.current, "minTrackingConfidence", {
          min: 0.1,
          max: 1.0,
          step: 0.1,
          label: "Tracking Sensitivity:",
        })
        .on("change", () => {
          updatePoseLandmarkerParams();
        });

      f2.addBinding(PARAMS.current, "frontHugValue", {
        min: 0.01,
        max: 1,
        step: 0.01,
        label: "Front Hug Tolerance:",
      });

      f2.addBinding(PARAMS.current, "wristsBelowShouldersThreshold", {
        min: -0.5,
        max: 0.5,
        step: 0.01,
        label: "Wrists Below Shoulders Threshold:",
      });

      f2.addBinding(PARAMS.current, "wristAboveElbowThreshold", {
        min: -0.5,
        max: 0.5,
        step: 0.01,
        label: "Wrist Above Elbow Threshold:",
      });

      f2.addBinding(PARAMS.current, "shoulderAboveHipThreshold", {
        min: 0.0,
        max: 1.0,
        step: 0.01,
        label: "Shoulder Above Hip Threshold:",
      });

      f2.addBinding(PARAMS.current, "wristAboveHipThreshold", {
        min: 0.0,
        max: 1.0,
        step: 0.01,
        label: "Wrist Above Hip Threshold:",
      });

      f2.addBinding(PARAMS.current, "sideHugShoulderThreshold", {
        min: 0.01,
        max: 0.5,
        step: 0.01,
        label: "Side Hug Shoulder Threshold:",
      });

      f2.addBinding(PARAMS.current, "sideHugElbowHipThreshold", {
        min: 0.01,
        max: 0.5,
        step: 0.01,
        label: "Side Hug Elbow-Hip Threshold:",
      });

      f2.addBinding(PARAMS.current, "elbowBentThreshold", {
        min: 90,
        max: 180,
        step: 1,
        label: "Elbow Bent Threshold:",
      });

      f2.addBinding(PARAMS.current, "VolumeIncreaseHD", {
        min: 0,
        max: 100,
        step: 1,
        label: "Volume Increase %:",
      });

      f3
        .addBinding(PARAMS.current, "showVideo", {
          label: "Show Video",
        })
        .on("change", (ev) => {
          setShowVideo(ev.value);
        });

      f3
        .addBinding(PARAMS.current, "showCanvas", {
          label: "Show Canvas",
        })
        .on("change", (ev) => {
          setShowCanvas(ev.value);
        });

      // Hide Tweakpane from the user by setting its display style to 'none'
      const paneElement = pane.element;
      if (paneElement) {
        paneElement.style.display = "none"; // This hides the Tweakpane UI
      }
    }
  }, [updatePoseLandmarkerParams]);

  // Initialize pose landmarker and audio playback after user interaction
  useEffect(() => {
    if (isStarted) {
      audioRef.current?.play();
      startDetection();
    }
  }, [isStarted]);

  // Volume and opacity fade-in/out logic
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateVolumeAndOpacity = () => {
      const targetVolume =
        isHugging || isSideHugging ? PARAMS.current.VolumeIncreaseHD / 100 : 0;

      const currentVolume = audio.volume;

      const fadeSpeed = PARAMS.current.fadeSpeed;

      const newVolume =
        currentVolume + (targetVolume - currentVolume) * fadeSpeed;

      audio.volume = Math.max(0, Math.min(1, newVolume));

      PARAMS.current.Volume = audio.volume * 100;

      // Update the video opacity based on the audio volume
      setVideoOpacity(audio.volume);
    };

    const intervalId = setInterval(updateVolumeAndOpacity, 50);

    return () => clearInterval(intervalId);
  }, [isHugging, isSideHugging, PARAMS.current.VolumeIncreaseHD]);

  // Manage introductory text display based on detection
  useEffect(() => {
    if (isPersonDetected && !isHugging && !isSideHugging) {
      setShowIntroText(true);
    } else {
      setShowIntroText(false);
    }
  }, [isPersonDetected, isHugging, isSideHugging]);

  return (
    <div className="w-[100vw] h-[100vh] overflow-hidden flex flex-row">
      <div className="flex flex-col overflow-hidden">
        <div className="relative overflow-hidden w-[100vw] h-[100vh]">
          {!isStarted && (
            <div
              className="absolute top-0 left-0 w-full h-full flex items-center justify-center bg-black bg-opacity-75 z-50"
              onClick={() => setIsStarted(true)}
            >
              <button className="text-white text-2xl">
                Start Hugging Detector 🫂
              </button>
            </div>
          )}
          <audio
            ref={audioRef}
            src={audioFilePath}
            loop
            className="w-1/4 absolute z-50 m-2"
          />
          <video
            ref={videoRef}
            muted
            className="absolute top-0 left-0 w-[100vw] h-[100vh] object-cover"
            style={{
              transform: "scaleX(-1)",
              display: showVideo ? "block" : "none",
            }}
          />
          <video
            src="/movies/CCI.webm"
            autoPlay
            muted
            loop
            className="absolute top-0 left-0 w-[100vw] h-[100vh] object-cover"
            style={{
              opacity: videoOpacity,
            }}
          />
          <canvas
            ref={poseCanvasRef}
            className="absolute top-0 left-0 w-[100vw] h-[100vh] object-cover"
            style={{
              backgroundColor: "transparent",
              transform: "scaleX(-1)",
              display: showCanvas ? "block" : "none",
            }}
          />
          {/* Introductory Text with Fade-In Effect */}
          <div
            className="absolute top-0 left-0 w-full h-full flex items-center justify-center"
            style={{
              opacity: showIntroText ? 1 : 0,
              transition: "opacity 3s",
            }}
          >
            <p className="text-white text-3xl text-center px-4 font-gap-sans">
              To see this piece, you must embrace yourself or another.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ObjectAndPoseDetector;
