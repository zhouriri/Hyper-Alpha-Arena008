import { useMemo } from 'react'
import Workstation from './Workstation'
import type { CharacterState } from './pixelData/characters'

export interface TraderData {
  accountId: number
  accountName: string
  avatarPresetId: number | null
  equity: number | null
  unrealizedPnl: number | null
  positionCount: number
  state: CharacterState
}

interface TradingFloorProps {
  traders: TraderData[]
}

export default function TradingFloor({ traders }: TradingFloorProps) {
  const cols = useMemo(() => {
    const count = traders.length
    if (count <= 1) return 1
    if (count <= 2) return 2
    if (count <= 4) return 2
    return 3
  }, [traders.length])

  return (
    <div className="relative w-full h-full min-h-[300px] rounded-lg overflow-hidden border border-border/30">
      {/* Dark floor background */}
      <div
        className="absolute inset-0"
        style={{
          background: 'linear-gradient(160deg, #0c0e14 0%, #111318 40%, #0e1016 100%)',
        }}
      />
      {/* Subtle floor tile pattern */}
      <div
        className="absolute inset-0 opacity-[0.035]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(150,180,220,0.8) 1px, transparent 1px),
            linear-gradient(90deg, rgba(150,180,220,0.8) 1px, transparent 1px)
          `,
          backgroundSize: '48px 48px',
        }}
      />
      {/* Ambient floor glow */}
      <div
        className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[70%] h-[50%] opacity-[0.03] rounded-full blur-3xl pointer-events-none"
        style={{ background: 'radial-gradient(ellipse, #6366f1, transparent)' }}
      />

      {/* Workstation grid */}
      <div className="relative z-10 flex items-center justify-center h-full p-6">
        <div
          className="grid"
          style={{
            gridTemplateColumns: `repeat(${cols}, 160px)`,
            gap: '24px 32px',
          }}
        >
          {traders.map((trader) => (
            <div key={trader.accountId} className="flex justify-center">
              <Workstation
                traderName={trader.accountName}
                equity={trader.equity}
                unrealizedPnl={trader.unrealizedPnl}
                positionCount={trader.positionCount}
                avatarPresetId={trader.avatarPresetId}
                state={trader.state}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
