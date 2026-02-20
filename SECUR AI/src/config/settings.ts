// frontend/src/config/settings.ts

export const RPI_CONFIG = {
  // 1. DATA API (server.py)
  // Used for: Login, Saving New Users, Approving Requests
  API_URL: "http://192.168.0.15:5001", //server.py http://192.168.0.15:5001

  // 2. SECURITY VIDEO FEED (main_main.py)
  // Used for: Admin Dashboard Live View, User Access Unlock
  VIDEO_FEED_URL: "http://192.168.0.15:5000/video_feed", //main_main.py http://192.168.0.15:5000/video_feed
  
  // 3. ENROLLMENT VIDEO FEED (app.py /)
  // Used for: New User Registration Page (Capture Step)
  ENROLLMENT_URL: "http://192.168.0.15:5002", //app.py http://192.168.0.15:5002

  // System Configuration
  IP_ADDRESS: "192.168.0.15",
  STATUS_CHECK_TIMEOUT: 5000,
  STATUS_CHECK_INTERVAL: 10000,
};

// Check if the Backend API (server.py) is online
export const checkRPiStatus = async (): Promise<boolean> => {
  try {
    const controller = new AbortController();
    // Set a short timeout (e.g. 3 seconds) so the UI updates quickly if offline
    const timeoutId = setTimeout(() => controller.abort(), RPI_CONFIG.STATUS_CHECK_TIMEOUT);
    
    // We explicitly check Port 5000's status endpoint
    // We construct the URL by taking the base of the VIDEO_FEED_URL
    const baseUrl = "http://192.168.0.15:5000"; 
    
    // The /status endpoint returns JSON like {"status": "IDLE", ...}
    const response = await fetch(`${baseUrl}/status`, {
      method: 'GET',
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    // If the fetch fails (timeout or connection refused), the system is OFFLINE
    return false;
  }
};