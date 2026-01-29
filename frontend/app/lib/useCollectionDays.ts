import { useState, useEffect } from 'react'

let cachedDays: number | null = null
let fetchPromise: Promise<number> | null = null

export function useCollectionDays() {
  const [days, setDays] = useState<number | null>(cachedDays)

  useEffect(() => {
    if (cachedDays !== null) {
      setDays(cachedDays)
      return
    }

    if (!fetchPromise) {
      fetchPromise = fetch('/api/system/collection-days')
        .then(res => res.json())
        .then(data => {
          cachedDays = data.days || 0
          return cachedDays
        })
        .catch(() => {
          cachedDays = 0
          return 0
        })
    }

    fetchPromise.then(d => setDays(d))
  }, [])

  return days
}
