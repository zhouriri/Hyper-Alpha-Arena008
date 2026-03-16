import { useEffect, useState } from 'react'
import { getPreset, SPRITE_FRAME_SIZE, SPRITE_COLS, DIR_DOWN } from './pixelData/palettes'
import type { CharacterState } from './pixelData/characters'

interface PixelCharacterProps {
  presetId: number | null
  state: CharacterState
  scale?: number  // display scale multiplier (e.g. 2 = 64x64)
}

// Walk animation frame columns: 0 -> 1 -> 2 -> 1 (loop)
const WALK_SEQUENCE = [0, 1, 2, 1]

export default function PixelCharacter({
  presetId,
  state,
  scale = 2.5,
}: PixelCharacterProps) {
  const preset = getPreset(presetId)
  const [frameIdx, setFrameIdx] = useState(1) // start at idle (middle frame)

  const isAnimating = state !== 'offline' && state !== 'idle'

  useEffect(() => {
    if (!isAnimating) {
      setFrameIdx(1) // idle = middle frame
      return
    }
    let idx = 0
    const timer = setInterval(() => {
      idx = (idx + 1) % WALK_SEQUENCE.length
      setFrameIdx(WALK_SEQUENCE[idx])
    }, 400)
    return () => clearInterval(timer)
  }, [isAnimating])

  const displaySize = SPRITE_FRAME_SIZE * scale
  const col = frameIdx
  const row = DIR_DOWN  // face toward viewer

  return (
    <div
      style={{
        width: displaySize,
        height: displaySize,
        backgroundImage: `url(/static/arena-sprites/${preset.sprite})`,
        backgroundSize: `${SPRITE_COLS * displaySize}px ${4 * displaySize}px`,
        backgroundPosition: `-${col * displaySize}px -${row * displaySize}px`,
        imageRendering: 'pixelated',
      }}
    />
  )
}
