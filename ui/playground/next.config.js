/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  env: {
    PERCEPTION_API_URL: process.env.NEXT_PUBLIC_PERCEPTION_API_URL || 'http://localhost:8080',
    RECONSTRUCTION_API_URL: process.env.NEXT_PUBLIC_RECONSTRUCTION_API_URL || 'http://localhost:8081',
  },
}
module.exports = nextConfig
