import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
    define:{
      'global': 'window',
    },
    server: {
        watch: {
            usePolling: true,
        },
        host: true,
        strictPort: true,
        port: 5173,

        allowedHosts: [
            "hygrophytic-aniyah-wretched.ngrok-free.dev"
        ],
        proxy: {
            '/ws': {
                target: 'http://localhost:8080', // Gira tutto al backend locale
                ws: true,                        // Importante per i WebSocket
                changeOrigin: true
            }
        }
    }

})
