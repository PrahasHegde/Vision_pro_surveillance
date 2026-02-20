import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ShieldCheck, User, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import FaceLockIcon from "@/components/FaceLockIcon";
import ThemeToggle from "@/components/ThemeToggle";
import AnimatedBackground from "@/components/AnimatedBackground";
import { RPI_CONFIG, checkRPiStatus } from "@/config/settings";

const Index = () => {
  const navigate = useNavigate();
  const [isOnline, setIsOnline] = useState<boolean | null>(null);

  useEffect(() => {
    // Check initial status
    checkRPiStatus().then(setIsOnline);

    // Set up interval to check status
    const interval = setInterval(() => {
      checkRPiStatus().then(setIsOnline);
    }, RPI_CONFIG.STATUS_CHECK_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  const buttons = [
    {
      label: "Admin",
      icon: ShieldCheck,
      onClick: () => navigate("/admin-login"),
      description: "System management & monitoring",
    },
    {
      label: "User",
      icon: User,
      onClick: () => navigate("/user"),
      description: "Access with face recognition",
    },
    {
      label: "New User",
      icon: UserPlus,
      onClick: () => navigate("/new-user"),
      description: "Register your face profile",
    },
  ];

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6 relative overflow-hidden">
      <AnimatedBackground />
      <ThemeToggle />
      
      {/* Subtle background glow effects */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/3 rounded-full blur-[100px]" />
      <div className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-primary/5 rounded-full blur-[80px]" />

      <div className="relative z-10 text-center max-w-lg mx-auto">
        <FaceLockIcon />
        
        <h1 className="text-4xl md:text-5xl font-bold mb-3 text-gradient">
          SECUR AI
        </h1>
        <p className="text-muted-foreground text-lg mb-12">
          "Be Smart. Stay Secure."
        </p>

        <div className="flex flex-col gap-4">
          {buttons.map((btn, index) => (
            <Button
              key={btn.label}
              // variant="default"
              onClick={btn.onClick}
              className="hero-button h-auto py-5 px-8 text-lg rounded-xl group w-full"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-center gap-4 w-full">
                <div className="p-2 rounded-lg bg-primary/20 group-hover:bg-primary-foreground/20 transition-colors">
                  <btn.icon className="w-6 h-6" />
                </div>
                <div className="text-left">
                  <div className="font-semibold">{btn.label}</div>
                  <div className="text-sm opacity-70 font-normal">
                    {btn.description}
                  </div>
                </div>
              </div>
            </Button>
          ))}
        </div>

        {/* System Status */}
        <div className="mt-12 text-xs text-muted-foreground font-mono">
          SYSTEM STATUS:{" "}
          {isOnline === null ? (
            <span className="text-amber-400">● CHECKING...</span>
          ) : isOnline ? (
            <span className="text-success">● ONLINE</span>
          ) : (
            <span className="text-destructive">● OFFLINE</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default Index;
