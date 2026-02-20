import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Camera, User, CheckCircle, Loader2, Mail, Calendar, Lock, Eye, EyeOff, Home, AtSign, Video, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { RPI_CONFIG } from "@/config/settings";

type RegistrationStep = "details" | "capture" | "processing" | "complete";

interface UserFormData {
  name: string;
  email: string;
  username: string;
  dob: string;
  password: string;
  confirmPassword: string;
}

const NewUser = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [step, setStep] = useState<RegistrationStep>("details");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  
  // --- Recording State ---
  const [isRecording, setIsRecording] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  const [timeLeft, setTimeLeft] = useState(30); 
  
  // --- Error States ---
  const [usernameError, setUsernameError] = useState("");
  const [emailError, setEmailError] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [formData, setFormData] = useState<UserFormData>({
    name: "", email: "", username: "", dob: "", password: "", confirmPassword: "",
  });

  // --- CAPTCHA & TERMS STATE ---
  const [num1, setNum1] = useState(0);
  const [num2, setNum2] = useState(0);
  const [operator, setOperator] = useState('+');
  const [captchaAnswer, setCaptchaAnswer] = useState('');
  const [isCaptchaVerified, setIsCaptchaVerified] = useState(false);
  const [captchaError, setCaptchaError] = useState(false);
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  // --- NEW: MODAL STATES ---
  const [showTerms, setShowTerms] = useState(false);
  const [showPrivacy, setShowPrivacy] = useState(false);

  // Generate a random math problem
  const generateCaptcha = () => {
    const n1 = Math.floor(Math.random() * 10) + 1;
    const n2 = Math.floor(Math.random() * 10) + 1;
    const ops = ['+', '-', '×'];
    const op = ops[Math.floor(Math.random() * ops.length)];

    // Prevent negative answers for subtraction
    if (op === '-' && n1 < n2) {
      setNum1(n2); setNum2(n1);
    } else {
      setNum1(n1); setNum2(n2);
    }
    setOperator(op);
    setCaptchaAnswer('');
    setIsCaptchaVerified(false);
    setCaptchaError(false);
  };

  // Run once when the component loads
  useEffect(() => {
    generateCaptcha();
  }, []);

  // Verify the user's answer
  const verifyCaptcha = (e: React.MouseEvent) => {
    e.preventDefault();
    let expected = 0;
    if (operator === '+') expected = num1 + num2;
    else if (operator === '-') expected = num1 - num2;
    else if (operator === '×') expected = num1 * num2;

    if (parseInt(captchaAnswer) === expected) {
      setIsCaptchaVerified(true);
      setCaptchaError(false);
    } else {
      setCaptchaError(true);
      setIsCaptchaVerified(false);
    }
  };

  // --- 1. FORM HANDLING ---
  const handleInputChange = (field: keyof UserFormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (field === "username") setUsernameError("");
    if (field === "email") setEmailError("");
  };

  const handleDetailsSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Basic Validation
    if (!formData.name.trim() || !formData.email.trim() || !formData.username.trim() || !formData.password) {
      toast({ title: "Error", description: "Please fill all fields", variant: "destructive" });
      return;
    }
    if (formData.password !== formData.confirmPassword) {
      toast({ title: "Error", description: "Passwords do not match", variant: "destructive" });
      return;
    }
    if (!isCaptchaVerified || !agreedToTerms) {
      toast({ title: "Error", description: "Please complete the verification and agree to terms", variant: "destructive" });
      return;
    }

    // --- CHECK AVAILABILITY (Username AND Email) ---
    try {
      const res = await fetch(`${RPI_CONFIG.API_URL}/check-availability`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            username: formData.username,
            email: formData.email 
        })
      });
      const data = await res.json();
      
      let hasError = false;

      if (!data.username_available) {
        setUsernameError("This username is already taken.");
        hasError = true;
      }
      
      if (!data.email_available) {
        setEmailError("This email is already registered.");
        hasError = true;
      }

      if (hasError) {
        toast({ title: "Registration Error", description: "Please fix the highlighted errors.", variant: "destructive" });
        return;
      }

      // If available, proceed to capture
      setStep("capture");

    } catch (error) {
       console.error("API Check failed", error);
       toast({ title: "Connection Error", description: "Could not verify details. Please check connection.", variant: "destructive" });
    }
  };

  // --- 2. CAMERA & RECORDING LOGIC ---
  const faceInstructions = [
    "Look straight at the camera",    
    "Turn your head slowly Left",     
    "Turn your head slowly Right",    
    "Look slightly Up",               
    "Look slightly Down",             
    "Smile naturally"                 
  ];
  const [currentInstruction, setCurrentInstruction] = useState(0);

  useEffect(() => {
    let stream: MediaStream | null = null;
    if (step === "capture") {
      navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480, facingMode: "user" }, 
        audio: false 
      })
      .then(s => {
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(err => {
        console.error("Camera Error:", err);
        setCameraError(true);
        toast({ title: "Camera Error", description: "Could not access webcam.", variant: "destructive" });
      });
    }
    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, [step]);

  const handleStartRecording = () => {
    if (!videoRef.current || !videoRef.current.srcObject) return;

    const stream = videoRef.current.srcObject as MediaStream;
    
    // --- FIX: FORCE VP8 CODEC FOR RASPBERRY PI COMPATIBILITY ---
    const options = { mimeType: 'video/webm; codecs=vp8' };
    let mediaRecorder: MediaRecorder;

    if (MediaRecorder.isTypeSupported(options.mimeType)) {
      mediaRecorder = new MediaRecorder(stream, options);
    } else {
      console.warn("VP8 not supported, falling back to default");
      mediaRecorder = new MediaRecorder(stream);
    }
    
    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    mediaRecorder.onstop = handleUploadVideo;

    mediaRecorder.start();
    setIsRecording(true);
    setCameraError(false);
    setCurrentInstruction(0);
    setTimeLeft(30); 
    
    const countdownInterval = setInterval(() => {
        setTimeLeft((prev) => {
            if (prev <= 1) {
                clearInterval(countdownInterval);
                stopRecording();
                return 0;
            }
            return prev - 1;
        });
    }, 1000);

    const instructionInterval = setInterval(() => {
      setCurrentInstruction((prev) => {
        if (prev >= faceInstructions.length - 1) {
          clearInterval(instructionInterval);
          return prev;
        }
        return prev + 1;
      });
    }, 5000); 
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  // --- 3. UPLOAD LOGIC ---
  const handleUploadVideo = async () => {
    setStep("processing");
    
    const blob = new Blob(chunksRef.current, { type: 'video/webm' });
    const videoFile = new File([blob], "recording.webm", { type: 'video/webm' });
    
    const formDataUpload = new FormData();
    formDataUpload.append("video", videoFile);
    formDataUpload.append("user_id", formData.username);

    try {
        const videoRes = await fetch(`${RPI_CONFIG.ENROLLMENT_URL}/upload_video`, {
            method: "POST",
            body: formDataUpload
        });
        const videoData = await videoRes.json();

        if (videoData.status !== "success") {
            throw new Error(videoData.message || "Video upload failed");
        }

        const userPayload = {
            id: Date.now().toString(),
            name: formData.name,
            email: formData.email,
            username: formData.username,
            password: formData.password,
            dob: formData.dob,
            timestamp: new Date().toLocaleString(),
            status: "pending",
        };

        const dataRes = await fetch(`${RPI_CONFIG.API_URL}/register-request`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userPayload)
        });

        if (dataRes.ok) {
            setStep("complete");
            toast({
                title: "Request Sent",
                description: "Your registration is pending Admin approval.",
            });
        } else {
            throw new Error("Failed to save user details");
        }
    } catch (error: any) {
        console.error(error);
        toast({ title: "Registration Failed", description: error.message, variant: "destructive" });
        setStep("details");
    }
  };

  // --- 4. RENDER UI ---
  const renderStep = () => {
    switch (step) {
      case "details":
        return (
          <form onSubmit={handleDetailsSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label>Full Name *</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input required value={formData.name} onChange={e => handleInputChange("name", e.target.value)} placeholder="Enter your full name" className="pl-10 bg-secondary/50 border-border focus:border-primary" />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Email Address *</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input required type="email" value={formData.email} onChange={e => handleInputChange("email", e.target.value)} placeholder="Enter your email" className={`pl-10 bg-secondary/50 border-border focus:border-primary ${emailError ? "border-destructive" : ""}`} />
              </div>
              {emailError && <p className="text-xs text-destructive">{emailError}</p>}
            </div>

            <div className="space-y-2">
              <Label>Username *</Label>
              <div className="relative">
                <AtSign className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input required value={formData.username} onChange={e => handleInputChange("username", e.target.value)} placeholder="Choose a unique username" className={`pl-10 bg-secondary/50 border-border focus:border-primary ${usernameError ? "border-destructive" : ""}`} />
              </div>
              {usernameError && <p className="text-xs text-destructive">{usernameError}</p>}
            </div>

            <div className="space-y-2">
              <Label>Date of Birth *</Label>
              <div className="relative">
                <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input required type="date" value={formData.dob} onChange={e => handleInputChange("dob", e.target.value)} className="pl-10 bg-secondary/50 border-border focus:border-primary" />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Password *</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input required type={showPassword ? "text" : "password"} value={formData.password} onChange={e => handleInputChange("password", e.target.value)} placeholder="Create a password" className="pl-10 pr-10 bg-secondary/50 border-border focus:border-primary" />
                <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Confirm Password *</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input required type={showConfirmPassword ? "text" : "password"} value={formData.confirmPassword} onChange={e => handleInputChange("confirmPassword", e.target.value)} placeholder="Confirm your password" className="pl-10 pr-10 bg-secondary/50 border-border focus:border-primary" />
                <button type="button" onClick={() => setShowConfirmPassword(!showConfirmPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
                  {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* --- CAPTCHA SECTION --- */}
            <div className="relative z-20 border border-border rounded-lg p-4 space-y-4 bg-secondary/10 mt-6">
              <div className="flex items-center gap-2">
                <div className={`w-4 h-4 rounded-full border flex items-center justify-center ${isCaptchaVerified ? 'border-primary bg-primary' : 'border-muted-foreground'}`}>
                  {isCaptchaVerified && <div className="w-2 h-2 bg-background rounded-full" />}
                </div>
                <span className="text-sm font-semibold text-foreground">Confirm you are not a robot</span>
              </div>

              {!isCaptchaVerified ? (
                <div className="space-y-3 pl-6">
                  <div className="flex items-center justify-between">
                    <div className="bg-background border border-border px-4 py-2 rounded-md font-mono text-lg font-bold text-foreground select-none">
                      {num1} {operator} {num2} = ?
                    </div>
                    <button type="button" onClick={generateCaptcha} className="text-xs text-muted-foreground hover:text-primary underline">
                      Refresh
                    </button>
                  </div>
                  <div className="flex gap-2">
                    <Input
                      type="number"
                      value={captchaAnswer}
                      onChange={(e) => setCaptchaAnswer(e.target.value)}
                      placeholder="Your answer"
                      className="flex-1 bg-secondary/50 border-border focus:border-primary"
                    />
                    <Button
                      type="button"
                      onClick={verifyCaptcha}
                      className="bg-primary/20 text-primary border border-primary/50 hover:bg-primary hover:text-primary-foreground font-medium"
                    >
                      Verify
                    </Button>
                  </div>
                  {captchaError && <p className="text-xs text-red-500">Incorrect answer. Please try again.</p>}
                </div>
              ) : (
                <p className="text-sm text-green-500 font-medium pl-6">✓ Verification successful</p>
              )}
            </div>

            {/* --- TERMS SECTION --- */}
            <div 
              className="relative z-20 border border-border rounded-lg p-4 bg-secondary/10 flex items-start gap-3 mt-4 cursor-pointer hover:bg-secondary/20 transition-colors"
              onClick={() => setAgreedToTerms(!agreedToTerms)}
            >
              <div className={`mt-0.5 w-4 h-4 rounded-full border flex items-center justify-center flex-shrink-0 transition-colors ${agreedToTerms ? 'border-primary bg-primary' : 'border-muted-foreground'}`}>
                {agreedToTerms && <div className="w-2 h-2 bg-background rounded-full" />}
              </div>
              <span className="text-sm text-muted-foreground select-none">
                I agree to the{" "}
                <span 
                  className="text-primary hover:underline font-medium"
                  onClick={(e) => {
                    e.stopPropagation(); // Prevents checking the box when clicking text
                    setShowTerms(true);
                  }}
                >
                  Terms & Conditions
                </span>{" "}
                and{" "}
                <span 
                  className="text-primary hover:underline font-medium"
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowPrivacy(true);
                  }}
                >
                  Privacy Policy
                </span>
              </span>
            </div>

            <Button 
              type="submit" 
              disabled={!isCaptchaVerified || !agreedToTerms}
              className="relative z-20 w-full hero-button h-12 mt-6 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Continue to Face Registration
            </Button>
          </form>
        );

      case "capture":
        return (
          <div className="space-y-6">
            <div className="relative aspect-video max-w-sm mx-auto bg-black rounded-2xl overflow-hidden border border-border shadow-lg">
              {!cameraError ? (
                <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]" />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center flex-col">
                  <Camera className="w-12 h-12 text-destructive/50" />
                  <p className="text-xs text-muted-foreground mt-2">Camera Access Denied</p>
                </div>
              )}
              <div className="absolute inset-8 border-2 border-dashed border-primary/40 rounded-full pointer-events-none" />
              {isRecording && (
                <>
                    <div className="absolute top-4 left-1/2 -translate-x-1/2 px-3 py-1 bg-destructive/80 rounded-full flex items-center gap-2 animate-pulse">
                        <span className="w-2 h-2 bg-white rounded-full" />
                        <span className="text-xs text-white font-mono">REC {timeLeft}s</span>
                    </div>
                    <div className="absolute bottom-8 left-0 right-0 text-center">
                        <span className="bg-black/60 text-white px-4 py-2 rounded-lg text-lg font-semibold backdrop-blur-sm transition-all duration-300">
                            {faceInstructions[currentInstruction]}
                        </span>
                    </div>
                </>
              )}
            </div>
            <div className="text-center">
              {isRecording ? (
                <div className="space-y-2">
                   <div className="flex justify-center gap-2 mt-4">
                     {faceInstructions.map((_, i) => (
                       <div key={i} className={`w-2 h-2 rounded-full transition-all duration-300 ${i === currentInstruction ? 'bg-primary scale-125' : 'bg-muted'}`} />
                     ))}
                   </div>
                </div>
              ) : (
                <div className="space-y-4">
                   <p className="text-sm text-muted-foreground">We will record a 30-second video to learn your face.<br/>Follow the instructions on screen.</p>
                   <Button onClick={handleStartRecording} className="hero-button"><Video className="w-4 h-4 mr-2" /> Start Recording</Button>
                </div>
              )}
            </div>
          </div>
        );

      case "processing":
        return (
          <div className="text-center py-12">
            <Loader2 className="w-16 h-16 mx-auto mb-4 animate-spin text-primary" />
            <h2 className="text-xl font-bold">Uploading Data...</h2>
            <p className="text-muted-foreground mt-2">Sending your profile to the admin for approval.</p>
          </div>
        );

      case "complete":
        return (
          <div className="text-center py-12">
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-amber-500/20 border border-amber-500 flex items-center justify-center">
              <CheckCircle className="w-10 h-10 text-amber-400" />
            </div>
            <p className="text-lg font-medium text-foreground">Request Submitted</p>
            <p className="text-sm text-amber-400 mt-1 font-medium">Pending Admin Approval</p>
            <p className="text-sm text-muted-foreground mt-2 mb-6">Your request has been sent to the admin.</p>
            <Button onClick={() => navigate("/")} className="hero-button"><Home className="w-4 h-4 mr-2" /> Return Home</Button>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-background grid-bg flex flex-col items-center justify-center p-6 relative overflow-hidden">
      <div className="absolute bottom-1/3 right-1/3 w-80 h-80 bg-primary/5 rounded-full blur-3xl" />
      <div className="relative z-10 w-full max-w-md">
        <div className="flex items-center justify-between mb-8">
          <Button variant="ghost" onClick={() => (step === "details" ? navigate("/") : setStep("details"))} className="text-muted-foreground hover:text-foreground">
            <ArrowLeft className="w-4 h-4 mr-2" /> {step === "details" ? "Back to Home" : "Back"}
          </Button>
          {step !== "complete" && (
            <Button variant="ghost" onClick={() => navigate("/")} className="text-muted-foreground hover:text-foreground"><Home className="w-4 h-4 mr-2" /> Home</Button>
          )}
        </div>
        <div className="glass-card p-8 glow-border">
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold text-foreground">New User Registration</h1>
            <p className="text-muted-foreground text-sm mt-1">{step === "details" && "Enter your details"}{step === "capture" && "Register your face"}{step === "processing" && "Please wait..."}{step === "complete" && "All done!"}</p>
          </div>
          {step !== "complete" && (
            <div className="flex items-center justify-center gap-2 mb-6">
              {["details", "capture", "processing"].map((s, i) => (
                <div key={s} className={`w-2 h-2 rounded-full transition-colors ${step === s ? "bg-primary" : i < ["details", "capture", "processing"].indexOf(step) ? "bg-primary/50" : "bg-muted"}`} />
              ))}
            </div>
          )}
          {renderStep()}
        </div>
      </div>

      {/* --- MODALS FOR TERMS AND PRIVACY --- */}
      {(showTerms || showPrivacy) && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-background border border-border rounded-xl shadow-2xl w-full max-w-md overflow-hidden animate-in fade-in zoom-in duration-200">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-bold text-foreground">
                {showTerms ? "Terms & Conditions" : "Privacy Policy"}
              </h2>
              <button 
                onClick={() => { setShowTerms(false); setShowPrivacy(false); }}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="p-6 max-h-[60vh] overflow-y-auto space-y-4 text-sm text-muted-foreground text-left">
              {showTerms ? (
                <>
                  <p><strong className="text-foreground">1. Acceptance of Terms:</strong> By registering, you agree to abide by the rules set forth by the system administrator.</p>
                  <p><strong className="text-foreground">2. Access Control:</strong> Access to the premises is granted at the sole discretion of the admin. Registration does not guarantee immediate access.</p>
                  <p><strong className="text-foreground">3. Appropriate Use:</strong> You agree not to tamper with, bypass, or attempt to deceive the SECUR AI - Smart Face Lock system.</p>
                  <p><strong className="text-foreground">4. Account Security:</strong> You are responsible for maintaining the confidentiality of your password. Do not share your credentials.</p>
                  <p><strong className="text-foreground">5. Termination:</strong> The administrator reserves the right to revoke your access at any time without prior notice.</p>
                </>
              ) : (
                <>
                  <p><strong className="text-foreground">1. Data Collection:</strong> We collect your name, email, date of birth, and facial biometric data (via a short video) solely for authentication purposes.</p>
                  <p><strong className="text-foreground">2. Data Storage:</strong> Your facial data and personal information are stored securely on the local Raspberry Pi server. We do not sell or share your data with third parties.</p>
                  <p><strong className="text-foreground">3. Biometric Usage:</strong> The 30-second video is processed to extract your facial features. The raw video is securely deleted upon admin action (approval or denial) to save storage space.</p>
                  <p><strong className="text-foreground">4. Right to Deletion:</strong> You may request the administrator to delete your profile at any time. This will permanently remove all your associated biometric data and images from the system.</p>
                  <p><strong className="text-foreground">5. Security:</strong> Use of this system is at your own risk. While data is kept local, no physical or digital system is entirely immune to interference.</p>
                </>
              )}
            </div>
            
            <div className="p-4 border-t border-border flex justify-end bg-secondary/20">
              <Button onClick={() => { setShowTerms(false); setShowPrivacy(false); }} className="hero-button px-6">
                Understood
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NewUser;