import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Camera, Home, User, Lock, Eye, EyeOff, Mail, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { RPI_CONFIG } from "@/config/settings";

type PageStep = "login" | "view";

interface UserDetails {
  id: string;
  name: string;
  email: string;
  username: string;
  password: string;
  dob: string;
  registeredAt: string;
}

const UserAccess = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  
  // Login state
  const [pageStep, setPageStep] = useState<PageStep>("login");
  const [loginIdentifier, setLoginIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [loggedInUser, setLoggedInUser] = useState<UserDetails | null>(null);
  
  // Feed state
  const [cameraError, setCameraError] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoggingIn(true);

    // FIX: Use the API_URL from settings.ts instead of hardcoded IP
    try {
      const response = await fetch(`${RPI_CONFIG.API_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ identifier: loginIdentifier, password: password })
      });

      const data = await response.json();

      if (response.ok && data.status === "success") {
        setLoggedInUser(data.user);
        setPageStep("view"); // Go directly to view mode
        toast({
          title: "Login Successful",
          description: `Welcome back, ${data.user.name}!`,
        });
      } else {
        toast({
          title: "Login Failed",
          description: "Invalid credentials or account not approved yet.",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Connection Error",
        description: "Could not connect to the server.",
        variant: "destructive",
      });
    }
    
    setIsLoggingIn(false);
  };

  const handleLogout = () => {
    setPageStep("login");
    setLoggedInUser(null);
    setLoginIdentifier("");
    setPassword("");
    setCameraError(false);
  };

  // Login Page UI
  if (pageStep === "login") {
    return (
      <div className="min-h-screen bg-background grid-bg flex flex-col items-center justify-center p-6 relative overflow-hidden">
        <div className="absolute top-1/3 left-1/3 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />

        <div className="relative z-10 w-full max-w-md">
          <Button
            variant="ghost"
            onClick={() => navigate("/")}
            className="mb-8 text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>

          <div className="glass-card p-8 glow-border">
            <div className="text-center mb-8">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/50 flex items-center justify-center pulse-ring-container">
                <User className="w-8 h-8 text-primary" />
                <div className="pulse-ring" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">User Login</h1>
              <p className="text-muted-foreground text-sm mt-1">
                Sign in to access security feed
              </p>
            </div>

            <form onSubmit={handleLogin} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="identifier" className="text-foreground">
                  Username or Email
                </Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    id="identifier"
                    type="text"
                    value={loginIdentifier}
                    onChange={(e) => setLoginIdentifier(e.target.value)}
                    placeholder="Enter username or email"
                    className="pl-10 bg-secondary/50 border-border focus:border-primary focus:ring-primary/20"
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-foreground">
                  Password
                </Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter password"
                    className="pl-10 pr-10 bg-secondary/50 border-border focus:border-primary focus:ring-primary/20"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showPassword ? (
                      <EyeOff className="w-4 h-4" />
                    ) : (
                      <Eye className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>

              <Button
                type="submit"
                className="w-full hero-button h-12 text-base"
                disabled={isLoggingIn}
              >
                {isLoggingIn ? (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                    Signing In...
                  </div>
                ) : (
                  "Sign In"
                )}
              </Button>
            </form>

            <p className="mt-6 text-center text-sm text-muted-foreground relative z-10">
              Don't have an account?{" "}
              <button
                type="button"
                onClick={() => navigate("/new-user")}
                className="text-primary hover:underline font-medium cursor-pointer bg-transparent border-none p-0 inline focus:outline-none"
              >
                Register Here
              </button>
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Live Feed View (Simplified, No Scanning)
  return (
    <div className="min-h-screen bg-background grid-bg flex flex-col items-center justify-center p-6 relative overflow-hidden">
      <div className="absolute top-1/3 left-1/3 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />

      <div className="relative z-10 w-full max-w-4xl"> {/* Increased width for better video view */}
        <div className="flex items-center justify-between mb-6">
          <Button
            variant="ghost"
            onClick={handleLogout}
            className="text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Logout
          </Button>
          <Button
            variant="ghost"
            onClick={() => navigate("/")}
            className="text-muted-foreground hover:text-foreground"
          >
            <Home className="w-4 h-4 mr-2" />
            Home
          </Button>
        </div>

        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-6">
             <div className="p-2 rounded-lg bg-primary/20 border border-primary/50">
                <ShieldCheck className="w-6 h-6 text-primary" />
             </div>
             <div>
                <h1 className="text-xl font-bold text-foreground">Live Security Feed</h1>
                <p className="text-sm text-muted-foreground">User: {loggedInUser?.name}</p>
             </div>
             <div className="ml-auto flex items-center gap-1.5 text-xs font-mono">
                <span className={`w-2 h-2 rounded-full ${cameraError ? 'bg-destructive' : 'bg-success animate-pulse'}`} />
                {cameraError ? 'OFFLINE' : 'LIVE'}
             </div>
          </div>

          {/* Camera Feed Container - Matches Admin Dashboard Style */}
          <div className="relative aspect-video bg-secondary/50 rounded-lg overflow-hidden border border-border">
            {!cameraError ? (
              <img 
                src={RPI_CONFIG.VIDEO_FEED_URL}
                alt="Live Camera Feed"
                className="absolute inset-0 w-full h-full object-contain"
                onError={() => setCameraError(true)}
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                    <div className="w-24 h-24 mx-auto mb-4 rounded-full border-2 border-dashed border-destructive/50 flex items-center justify-center">
                      <Camera className="w-10 h-10 text-destructive/50" />
                    </div>
                    <p className="text-destructive text-sm font-medium">Camera Feed Unavailable</p>
                    <button 
                      onClick={() => setCameraError(false)}
                      className="mt-3 px-4 py-1.5 text-xs bg-primary/20 text-primary border border-primary/30 rounded hover:bg-primary/30 transition-colors"
                    >
                      Retry Connection
                    </button>
                </div>
              </div>
            )}
            
            {/* Timestamp overlay */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1 bg-black/60 rounded text-xs font-mono text-foreground">
                LIVE VIEW
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserAccess;