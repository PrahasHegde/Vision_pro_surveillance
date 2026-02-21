# 👁️ Frontend - SECUR AI Computer Vision Based Smart Face Lock

This is the React-based frontend interface for the Computer Vision Based Smart Face Lock system. It provides a secure, modern, and responsive dashboard for administrators to monitor access, manage users, and handle new biometric enrollments directly from a web browser or mobile device.

## ✨ Key Features

* **Admin Dashboard:** View the live stereo-vision camera feed, monitor real-time activity logs, and check the system's online/offline status.
* **User Management:** View registered users, approve or deny pending registration requests, and securely delete users (automatically synchronizing with the backend dataset and AI models).
* **Remote Unlock:** A manual override "Hold to Unlock" button for administrators to trigger the physical door lock remotely.
* **Secure Biometric Enrollment:** * Multi-step registration flow for new users.
  * Dynamic Math CAPTCHA for bot protection.
  * Required Terms & Conditions and Privacy Policy agreements with interactive modals.
  * In-browser 30-second video capture (encoded in VP8 WebM for maximum Raspberry Pi compatibility) to securely gather training data for facial recognition.
* **Responsive UI:** Built with Tailwind CSS and glass-morphism effects for a sleek, modern look that scales perfectly from desktop monitors to smartphone screens.

## 🛠️ Tech Stack

* **Framework:** [React](https://reactjs.org/) + [TypeScript](https://www.typescriptlang.org/)
* **Build Tool:** [Vite](https://vitejs.dev/)
* **Routing:** React Router v6
* **Styling:** [Tailwind CSS](https://tailwindcss.com/)
* **UI Components:** [Radix UI](https://www.radix-ui.com/) / [shadcn/ui](https://ui.shadcn.com/)
* **Icons:** [Lucide React](https://lucide.dev/)

## ⚙️ Configuration (`settings.ts`)

Before running or building the project, you must configure the backend connection URLs so that the frontend knows how to communicate with the Raspberry Pi. Open `src/config/settings.ts` and set your URLs based on your environment.

### For Local Network Testing (Wi-Fi)
If you are developing locally and connected to the same Wi-Fi as the Raspberry Pi:
```typescript
export const RPI_CONFIG = {
  API_URL: "[http://192.168.0.15:5001](http://192.168.0.15:5001)",
  VIDEO_FEED_URL: "[http://192.168.0.15:5000/video_feed](http://192.168.0.15:5000/video_feed)",
  ENROLLMENT_URL: "[http://192.168.0.15:5002](http://192.168.0.15:5002)",
  IP_ADDRESS: "192.168.0.15",
  STATUS_CHECK_TIMEOUT: 5000,
  STATUS_CHECK_INTERVAL: 10000,
};

### For Production (Cloudflare Tunnels)
When accessing the system securely over the internet (4G/5G), update the URLs to your generated Cloudflare Tunnel links. **Note: Cloudflare provides HTTPS automatically.**

```typescript
export const RPI_CONFIG = {
  API_URL: "https://<your-api-tunnel>.trycloudflare.com",
  // CRITICAL: Ensure /video_feed remains at the end of the video URL!
  VIDEO_FEED_URL: "https://<your-video-tunnel>[.trycloudflare.com/video_feed](https://.trycloudflare.com/video_feed)", 
  ENROLLMENT_URL: "https://<your-upload-tunnel>.trycloudflare.com",
  IP_ADDRESS: "192.168.0.15", 
  STATUS_CHECK_TIMEOUT: 5000,
  STATUS_CHECK_INTERVAL: 10000,
};

## 🚀 Local Development Setup

1. **Install dependencies:**
   ```bash
   npm install

Start the local development server:

Bash
npm run dev
