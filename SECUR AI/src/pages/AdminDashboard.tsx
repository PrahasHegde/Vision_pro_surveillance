import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Home, Camera, Check, X, Clock, User, ShieldCheck, AlertCircle,
  Calendar, Mail, Unlock, Users, Trash2, ChevronDown, ChevronUp,
  PlayCircle, Video // <--- Added Icons
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { RPI_CONFIG } from "@/config/settings";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";

// --- INTERFACES ---
interface RegisteredUser {
  id: string;
  name: string;
  email: string;
  username: string;
  dob: string;
  registeredAt: string;
}

interface NewUserRequest {
  id: string;
  name: string;
  email: string;
  username: string;
  dob: string;
  timestamp: string;
  status: "pending";
}

interface LogEntry {
  id: string;
  name: string;
  action: "granted" | "denied" | "detected" | "manual";
  timestamp: string;
  date: string;
  confidence?: number;
}

const AdminDashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  
  // Camera & Unlock State
  const [cameraError, setCameraError] = useState(false);
  const [isUnlocking, setIsUnlocking] = useState(false);
  const [holdProgress, setHoldProgress] = useState(0);
  const [holdTimer, setHoldTimer] = useState<NodeJS.Timeout | null>(null);

  // Data State
  const [accessRequests, setAccessRequests] = useState<NewUserRequest[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [registeredUsers, setRegisteredUsers] = useState<RegisteredUser[]>([]);
  const [expandedUserId, setExpandedUserId] = useState<string | null>(null);
  const [userToDelete, setUserToDelete] = useState<RegisteredUser | null>(null);

  // --- NEW: VIDEO PLAYER STATE ---
  const [selectedVideo, setSelectedVideo] = useState<{url: string, user: string} | null>(null);

  // --- FETCH DATA ---
  const fetchDashboardData = async () => {
    try {
      // 1. Fetch Users & Requests (Added cache buster: ?_=${Date.now()})
      const responseApi = await fetch(`${RPI_CONFIG.API_URL}/get-all-data?_=${Date.now()}`);
      if (responseApi.ok) {
        const data = await responseApi.json();
        setRegisteredUsers(data.users || []);
        setAccessRequests(data.pending || []);
      }

      // 2. Fetch Logs from Main System (Added cache buster)
      const videoUrlObj = new URL(RPI_CONFIG.VIDEO_FEED_URL);
      const mainSystemUrl = videoUrlObj.origin; 
      
      const responseLogs = await fetch(`${mainSystemUrl}/logs?_=${Date.now()}`);
      if (responseLogs.ok) {
        const rawLogs = await responseLogs.json();
        
        const formattedLogs: LogEntry[] = rawLogs.map((log: any, index: number) => {
          const [datePart, timePart] = log.timestamp.split(' ');
          let actionType: LogEntry['action'] = 'detected';
          if (log.action.includes("GRANTED")) actionType = 'granted';
          else if (log.action.includes("DENIED")) actionType = 'denied';
          else if (log.action.includes("MANUAL")) actionType = 'manual';

          return {
            id: `log-${index}`,
            name: log.user,
            action: actionType,
            timestamp: timePart,
            date: datePart
          };
        });
        setLogs(formattedLogs);
      }
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const pollInterval = setInterval(fetchDashboardData, 600000); 
    return () => clearInterval(pollInterval);
  }, []);

  const handleUnlock = async () => {
    setIsUnlocking(true);
    setHoldProgress(0);
    try {
        const videoUrlObj = new URL(RPI_CONFIG.VIDEO_FEED_URL);
        const baseUrl = videoUrlObj.origin; 
        
        toast({ title: "Sending Command...", description: "Requesting door unlock..." });

        const response = await fetch(`${baseUrl}/unlock`, { method: 'POST' });
        const data = await response.json();

        if (response.ok && data.status === "success") {
            toast({
                title: "Door Unlocked",
                description: "The door has been successfully opened.",
                className: "bg-green-600 text-white border-none", 
            });
            fetchDashboardData();
        } else {
            throw new Error(data.message || "Failed to unlock");
        }
    } catch (error: any) {
        toast({ title: "Unlock Failed", description: error.message, variant: "destructive" });
    } finally {
        setIsUnlocking(false);
    }
  };

  const handleApprove = async (id: string) => {
    try {
      const request = accessRequests.find((r) => r.id === id);
      if (!request) return;

      const response = await fetch(`${RPI_CONFIG.API_URL}/approve-user`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: id, registeredAt: new Date().toISOString() }),
      });

      if (response.ok) {
        toast({ title: "User Approved", description: `${request.name} is now active.` });
        fetchDashboardData(); 
        setSelectedVideo(null); // Close video if open
      }
    } catch (error) {
      toast({ title: "Error", description: "Could not approve user", variant: "destructive" });
    }
  };

  const handleDeny = async (id: string) => {
    try {
      const response = await fetch(`${RPI_CONFIG.API_URL}/deny-user`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: id }),
      });

      if (response.ok) {
        toast({ title: "User Denied", description: "Request removed.", variant: "destructive" });
        fetchDashboardData();
        setSelectedVideo(null); // Close video if open
      }
    } catch (error) {
      toast({ title: "Error", description: "Could not deny user", variant: "destructive" });
    }
  };

  const handleRemoveUser = async () => {
    if (!userToDelete) return;
    try {
      const response = await fetch(`${RPI_CONFIG.API_URL}/delete-user`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: userToDelete.id }),
      });

      if (response.ok) {
        toast({ title: "User Removed", description: `${userToDelete.name} deleted.`, variant: "destructive" });
        fetchDashboardData();
        setUserToDelete(null);
      }
    } catch (error) {
      toast({ title: "Error", description: "Could not delete user", variant: "destructive" });
    }
  };

  const handleUnlockStart = () => {
    if(isUnlocking) return;
    const startTime = Date.now();
    const timer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min((elapsed / 2000) * 100, 100); 
      setHoldProgress(progress);
      if (progress >= 100) {
        clearInterval(timer);
        setHoldTimer(null);
        handleUnlock(); 
      }
    }, 50);
    setHoldTimer(timer);
  };

  const handleUnlockEnd = () => {
    if (holdTimer) { clearInterval(holdTimer); setHoldTimer(null); }
    setHoldProgress(0);
  };

  const toggleUserExpand = (userId: string) => {
    setExpandedUserId(expandedUserId === userId ? null : userId);
  };

  const getActionStyle = (action: LogEntry["action"]) => {
    switch (action) {
      case "granted": return "text-green-500 bg-green-500/10 border-green-500/30";
      case "denied": return "text-red-500 bg-red-500/10 border-red-500/30";
      case "detected": return "text-amber-400 bg-amber-500/10 border-amber-500/30";
      case "manual": return "text-blue-400 bg-blue-500/10 border-blue-500/30";
      default: return "";
    }
  };

  // --- NEW: HELPER TO OPEN VIDEO ---
  const openVideo = (request: NewUserRequest) => {
    // Construct URL: http://<RPI_IP>:5001/videos/<id>.mp4
    // NOTE: Ensure your backend saves as .mp4!
    const url = `${RPI_CONFIG.API_URL}/videos/${request.username}.mp4`;
    setSelectedVideo({ url, user: request.name });
  };

  return (
    <div className="min-h-screen bg-background grid-bg p-4 md:p-6 relative">
      
      {/* --- NEW: VIDEO PLAYER MODAL --- */}
      {selectedVideo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="bg-zinc-900 border border-white/10 rounded-xl max-w-2xl w-full overflow-hidden shadow-2xl scale-100">
            <div className="flex items-center justify-between p-4 border-b border-white/10 bg-zinc-900/50">
              <div className="flex items-center gap-2">
                <Video className="w-5 h-5 text-primary" />
                <h3 className="font-semibold text-lg">Registration Video — {selectedVideo.user}</h3>
              </div>
              <button 
                onClick={() => setSelectedVideo(null)}
                className="p-1 hover:bg-white/10 rounded-full transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="p-4 bg-black">
              <video 
                src={selectedVideo.url} 
                controls 
                autoPlay 
                className="w-full h-auto rounded-lg border border-white/10"
                onError={(e) => toast({title: "Video Error", description: "Could not play video. Check format (.mp4 required).", variant: "destructive"})}
              />
            </div>

            <div className="p-4 bg-zinc-900/50 border-t border-white/10 flex justify-end gap-2">
               <Button variant="outline" onClick={() => setSelectedVideo(null)}>Close</Button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/20 border border-primary/50">
            <ShieldCheck className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">Admin Dashboard</h1>
            <p className="text-sm text-muted-foreground font-mono">System Monitor</p>
          </div>
        </div>
        <Button variant="outline" onClick={() => navigate("/")} className="border-border hover:bg-secondary">
          <Home className="w-4 h-4 mr-2" />
          Home
        </Button>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT COLUMN: Camera & Unlock Button */}
        <div className="lg:col-span-2 space-y-4">
          {/* Camera Feed Card */}
          <div className="glass-card p-4">
            <div className="flex items-center gap-2 mb-4">
              <Camera className="w-5 h-5 text-primary" />
              <h2 className="font-semibold text-foreground">Live Camera Feed</h2>
              <span className="ml-auto flex items-center gap-1.5 text-xs font-mono">
                <span className={`w-2 h-2 rounded-full ${cameraError ? 'bg-destructive' : 'bg-green-500 animate-pulse'}`} />
                {cameraError ? 'OFFLINE' : 'LIVE'}
              </span>
            </div>
            <div className="relative aspect-video bg-secondary/50 rounded-lg overflow-hidden border border-border">
              {!cameraError ? (
                <img 
                  src={RPI_CONFIG.VIDEO_FEED_URL}
                  alt="Live Camera Feed"
                  className="absolute inset-0 w-full h-full object-contain"
                  onError={() => setCameraError(true)}
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center flex-col gap-2">
                    <Camera className="w-10 h-10 text-muted-foreground/50" />
                    <p className="text-muted-foreground text-sm">Signal Lost</p>
                    <Button variant="outline" size="sm" onClick={() => setCameraError(false)}>Retry</Button>
                </div>
              )}
            </div>
          </div>

          {/* Hold to Unlock Button */}
          <div className="relative">
            <Button
              onMouseDown={handleUnlockStart}
              onMouseUp={handleUnlockEnd}
              onMouseLeave={handleUnlockEnd}
              onTouchStart={handleUnlockStart}
              onTouchEnd={handleUnlockEnd}
              disabled={isUnlocking}
              className="w-full h-16 text-lg font-bold bg-green-500/10 text-green-500 border-2 border-green-500/50 hover:bg-green-500/20 transition-all duration-300 relative overflow-hidden"
            >
              <div 
                className="absolute left-0 top-0 bottom-0 bg-green-500/30 transition-all duration-75 ease-linear"
                style={{ width: `${holdProgress}%` }}
              />
              <div className="relative z-10 flex items-center justify-center gap-3">
                <Unlock className={`w-6 h-6 ${isUnlocking ? 'animate-bounce' : ''}`} />
                {isUnlocking ? "Unlocking..." : holdProgress > 0 ? "HOLDING..." : "HOLD TO UNLOCK"}
              </div>
            </Button>
          </div>
        </div>

        {/* RIGHT COLUMN: Lists */}
        <div className="space-y-6">
          
          {/* 1. Pending Requests (UPDATED WITH VIDEO BUTTON) */}
          <div className="glass-card p-4">
            <div className="flex items-center gap-2 mb-4">
              <AlertCircle className="w-5 h-5 text-amber-400" />
              <h2 className="font-semibold text-foreground">Pending Requests</h2>
              {accessRequests.length > 0 && (
                <span className="ml-auto px-2 py-0.5 text-xs font-mono rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">
                  {accessRequests.length}
                </span>
              )}
            </div>
            <ScrollArea className="h-48">
              {accessRequests.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-6">No pending requests</p>
              ) : (
                <div className="space-y-3 pr-4">
                  {accessRequests.map((request) => (
                    <div key={request.id} className="p-3 rounded-lg bg-secondary/50 border border-border">
                      <div className="flex items-start gap-2 mb-2">
                        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                          <User className="w-4 h-4 text-primary" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-foreground truncate">{request.name}</p>
                          <p className="text-xs text-muted-foreground">@{request.username}</p>
                        </div>
                      </div>
                      
                      {/* --- NEW: View Video Button --- */}
                      <Button 
                        size="sm" 
                        variant="secondary"
                        onClick={() => openVideo(request)} 
                        className="w-full mb-2 h-8 bg-blue-500/10 text-blue-400 border border-blue-500/30 hover:bg-blue-500 hover:text-white"
                      >
                        <PlayCircle className="w-4 h-4 mr-2" /> View Recorded Video
                      </Button>
                      
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => handleApprove(request.id)} className="flex-1 h-8 bg-green-500/20 text-green-500 border border-green-500/30 hover:bg-green-500 hover:text-white">
                          <Check className="w-4 h-4 mr-1" /> Approve
                        </Button>
                        <Button size="sm" onClick={() => handleDeny(request.id)} className="flex-1 h-8 bg-red-500/20 text-red-500 border border-red-500/30 hover:bg-red-500 hover:text-white">
                          <X className="w-4 h-4 mr-1" /> Deny
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </div>

          {/* 2. Registered Users (Unchanged) */}
          <div className="glass-card p-4">
            <div className="flex items-center gap-2 mb-4">
              <Users className="w-5 h-5 text-primary" />
              <h2 className="font-semibold text-foreground">Registered Users</h2>
            </div>
            <ScrollArea className="h-56">
              {registeredUsers.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-6">No registered users</p>
              ) : (
                <div className="space-y-2 pr-4">
                  {registeredUsers.map((user) => (
                    <div key={user.id} className="rounded-lg bg-secondary/50 border border-border overflow-hidden">
                      <button onClick={() => toggleUserExpand(user.id)} className="w-full p-3 flex items-center gap-3 hover:bg-secondary/80 transition-colors">
                        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                          <User className="w-4 h-4 text-primary" />
                        </div>
                        <div className="flex-1 text-left min-w-0">
                          <p className="text-sm font-medium text-foreground truncate">{user.name}</p>
                          <p className="text-xs text-muted-foreground">@{user.username}</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button size="sm" variant="ghost" onClick={(e) => { e.stopPropagation(); setUserToDelete(user); }} className="h-8 w-8 p-0 text-destructive hover:bg-destructive/20 hover:text-destructive">
                            <Trash2 className="w-4 h-4" />
                          </Button>
                          {expandedUserId === user.id ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        </div>
                      </button>
                      {expandedUserId === user.id && (
                        <div className="px-3 pb-3 pt-1 border-t border-border/50 space-y-2">
                          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Mail className="w-3 h-3" /> {user.email}</div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground"><Calendar className="w-3 h-3" /> DOB: {user.dob}</div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </div>

          {/* 3. Activity Logs (Unchanged) */}
          <div className="glass-card p-4">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-primary" />
              <h2 className="font-semibold text-foreground">Activity Logs</h2>
            </div>
            <ScrollArea className="h-64">
              <div className="space-y-2 pr-4">
                {logs.map((log) => (
                  <div key={log.id} className="p-2.5 rounded-lg bg-secondary/30 border border-border/50 flex items-center gap-3">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">{log.name}</p>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground font-mono">{log.date} {log.timestamp}</span>
                      </div>
                    </div>
                    <span className={`px-2 py-0.5 text-xs font-medium rounded border capitalize ${getActionStyle(log.action)}`}>
                      {log.action}
                    </span>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!userToDelete} onOpenChange={() => setUserToDelete(null)}>
        <AlertDialogContent className="bg-background border-border">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-foreground">Remove User</AlertDialogTitle>
            <AlertDialogDescription className="text-muted-foreground">
              Are you sure you want to remove <span className="font-semibold text-foreground">{userToDelete?.name}</span>? 
              This will delete their data from the system database.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="border-border hover:bg-secondary">Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleRemoveUser} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Remove User
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default AdminDashboard;