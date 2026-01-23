'use client'

import { useState } from 'react'
import { supabase } from '@/lib/supabase'

export default function Home() {
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setMessage('')

    try {
      const fileName = `video/${Date.now()}_${file.name}`
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('videos')
        .upload(fileName, file)

      if (uploadError) {
        setMessage(`Error: ${uploadError.message}`)
      } else {
        // Notify Flask API about the upload
        const flaskResponse = await fetch('http://localhost:5001/notify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ video_path: uploadData.path })
        })
        const flaskData = await flaskResponse.json()
        setMessage(`Supabase: ${uploadData.path} | Flask: ${flaskData.message}`)
      }
    } catch (error) {
      setMessage(`Error: ${error}`)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-8">
      <div className="w-full max-w-md space-y-4">
        <h1 className="text-2xl font-bold">Upload Video</h1>
        <input
          type="file"
          accept="video/*"
          onChange={handleUpload}
          disabled={uploading}
          className="w-full"
        />
        {uploading && <p>Uploading...</p>}
        {message && <p className="text-sm">{message}</p>}
      </div>
    </div>
  )
}
