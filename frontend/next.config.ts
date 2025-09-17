import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Optimize for Deno runtime
  experimental: {
    turbo: {
      memoryLimit: 1536, // Reduced for Netlify constraints
    },
    optimizePackageImports: ['lucide-react'],
  },

  // Essential for Deno deployment
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },

  // Optimize for deployment
  poweredByHeader: false,
  compress: true,

  // Webpack optimization for memory and Deno
  webpack: (config, { isServer, dev }) => {
    // Memory optimization
    if (!dev) {
      config.optimization = {
        ...config.optimization,
        minimize: true,
        splitChunks: {
          chunks: 'all',
          minSize: 20000,
          maxSize: 244000,
          cacheGroups: {
            default: {
              minChunks: 2,
              priority: -20,
              reuseExistingChunk: true,
            },
            vendor: {
              test: /[\\/]node_modules[\\/]/,
              name: 'vendors',
              priority: -10,
              chunks: 'all',
              maxSize: 200000,
            },
          },
        },
      };
    }

    // Deno-specific optimizations
    config.resolve = {
      ...config.resolve,
      alias: {
        ...config.resolve.alias,
        // Optimize for Deno imports
        'react': 'npm:react',
        'react-dom': 'npm:react-dom',
      },
    };

    return config;
  },

  // Build optimizations
  typescript: {
    ignoreBuildErrors: false,
  },

  eslint: {
    ignoreDuringBuilds: false,
  },

  // Static generation for better compatibility
  generateStaticParams: true,
};

export default nextConfig;
