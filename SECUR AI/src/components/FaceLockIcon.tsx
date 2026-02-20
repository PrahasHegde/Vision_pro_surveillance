import { Scan } from "lucide-react";

const FaceLockIcon = () => {
  return (
    <div className="relative w-32 h-32 mx-auto mb-8">
      {/* Pulse rings */}
      <div className="pulse-ring" style={{ animationDelay: "0s" }} />
      <div className="pulse-ring" style={{ animationDelay: "0.5s" }} />
      <div className="pulse-ring" style={{ animationDelay: "1s" }} />
      
      {/* Main icon container */}
      <div className="absolute inset-4 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/50 flex items-center justify-center">
        <Scan className="w-12 h-12 text-primary" />
      </div>
    </div>
  );
};

export default FaceLockIcon;
