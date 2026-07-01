# Deploying FabricFind Backend to Railway

## Step 1 — Push backend to GitHub
1. Create a new GitHub repo called `fabricfind-backend-mobile`
2. Upload these files to it:
   - server.py
   - requirements.txt
   - Procfile
   - fabricfind_model.onnx
   - fabricfind_model.onnx.data

## Step 2 — Deploy to Railway
1. Go to https://railway.app
2. New Project → GitHub Repository → select fabricfind-backend-mobile
3. Railway auto-deploys

## Step 3 — Set Environment Variables in Railway
In your Railway project → Variables tab, add:

CLOUDINARY_CLOUD_NAME     = your cloud name from Cloudinary dashboard
CLOUDINARY_API_KEY        = your API key
CLOUDINARY_API_SECRET     = your API secret
FIREBASE_PROJECT_ID       = fabricfind-a8f7f
GOOGLE_APPLICATION_CREDENTIALS = serviceAccount.json

## Step 4 — Firebase Service Account
1. Go to Firebase Console → Project Settings → Service Accounts
2. Click "Generate new private key" → downloads serviceAccount.json
3. Upload serviceAccount.json to your GitHub repo (or use Railway's file upload)
   WARNING: Keep this file private — do not share publicly

## Step 5 — Verify
Visit your Railway URL: https://your-app.up.railway.app
Should return: { "status": "FabricFind Mobile Backend running" }

## Step 6 — Note your Railway URL
You'll need it for the mobile app:
BACKEND_URL = "https://your-app.up.railway.app"
