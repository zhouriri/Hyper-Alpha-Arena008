// Avatar presets for Arena View — maps preset IDs to sprite sheet files
// Sprites sourced from Stanford Generative Agents (MIT license)
// Each sprite sheet: 96x128 PNG, 3 cols x 4 rows, 32x32 per frame
// Row 0: face down, Row 1: face left, Row 2: face right, Row 3: face up

export interface AvatarPreset {
  id: number
  sprite: string  // filename in /static/arena-sprites/
  label: string   // character reference name
}

export const AVATAR_PRESETS: AvatarPreset[] = [
  { id: 1,  sprite: 'Klaus_Mueller.png',      label: 'Klaus' },
  { id: 2,  sprite: 'Carlos_Gomez.png',       label: 'Carlos' },
  { id: 3,  sprite: 'Sam_Moore.png',          label: 'Sam' },
  { id: 4,  sprite: 'Eddy_Lin.png',           label: 'Eddy' },
  { id: 5,  sprite: 'Arthur_Burton.png',      label: 'Arthur' },
  { id: 6,  sprite: 'Rajiv_Patel.png',        label: 'Rajiv' },
  { id: 7,  sprite: 'Isabella_Rodriguez.png', label: 'Isabella' },
  { id: 8,  sprite: 'Mei_Lin.png',            label: 'Mei' },
  { id: 9,  sprite: 'Hailey_Johnson.png',     label: 'Hailey' },
  { id: 10, sprite: 'Tamara_Taylor.png',      label: 'Tamara' },
  { id: 11, sprite: 'Jennifer_Moore.png',     label: 'Jennifer' },
  { id: 12, sprite: 'Yuriko_Yamamoto.png',    label: 'Yuriko' },
]

export function getPreset(id: number | null | undefined): AvatarPreset {
  if (!id) return AVATAR_PRESETS[0]
  return AVATAR_PRESETS.find(p => p.id === id) || AVATAR_PRESETS[0]
}

// Sprite sheet constants
export const SPRITE_FRAME_SIZE = 32  // each frame is 32x32
export const SPRITE_COLS = 3
export const SPRITE_ROWS = 4
// Row indices by direction
export const DIR_DOWN = 0
export const DIR_LEFT = 1
export const DIR_RIGHT = 2
export const DIR_UP = 3
