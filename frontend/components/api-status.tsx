'use client';

import React from 'react';
import { Globe, Server, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { isGradioEnabled, getGradioConfig } from '../lib/gradio-client';

interface ApiStatusProps {
  className?: string;
}

export default function ApiStatus({ className = '' }: ApiStatusProps) {
  const isUsingGradio = isGradioEnabled();
  const gradioConfig = getGradioConfig();

  return (
    <div className={`inline-flex items-center space-x-2 text-xs ${className}`}>
      {isUsingGradio ? (
        <>
          <Globe className="h-3 w-3 text-blue-500" />
          <span className="text-gray-600">Gradio</span>
          <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded-full font-medium">
            {gradioConfig.spaceId}
          </span>
        </>
      ) : (
        <>
          <Server className="h-3 w-3 text-green-500" />
          <span className="text-gray-600">Local API</span>
          <span className="px-2 py-1 bg-green-50 text-green-700 rounded-full font-medium">
            {process.env.NEXT_PUBLIC_API_BASE_URL || 'localhost:8000'}
          </span>
        </>
      )}
    </div>
  );
}

export function ApiStatusBadge({ className = '' }: ApiStatusProps) {
  const isUsingGradio = isGradioEnabled();
  const gradioConfig = getGradioConfig();

  return (
    <div className={`inline-flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${className}`}>
      {isUsingGradio ? (
        <>
          <div className="flex items-center space-x-1 bg-blue-50 text-blue-700 px-2 py-1 rounded-full">
            <Globe className="h-3 w-3" />
            <span>Gradio</span>
          </div>
        </>
      ) : (
        <>
          <div className="flex items-center space-x-1 bg-green-50 text-green-700 px-2 py-1 rounded-full">
            <Server className="h-3 w-3" />
            <span>Local</span>
          </div>
        </>
      )}
    </div>
  );
}
