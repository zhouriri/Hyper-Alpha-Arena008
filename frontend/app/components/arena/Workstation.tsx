import PixelCharacter from './PixelCharacter'
import TraderHUD from './TraderHUD'
import type { CharacterState } from './pixelData/characters'

interface WorkstationProps {
  traderName: string
  equity: number | null
  unrealizedPnl: number | null
  positionCount: number
  avatarPresetId: number | null
  state: CharacterState
}

const STATE_GLOW: Record<CharacterState, string> = {
  offline: 'transparent',
  error: 'rgba(239,68,68,0.12)',
  program_running: 'rgba(139,92,246,0.12)',
  just_traded: 'rgba(234,179,8,0.15)',
  ai_thinking: 'rgba(59,130,246,0.1)',
  holding_profit: 'rgba(34,197,94,0.12)',
  holding_loss: 'rgba(239,68,68,0.12)',
  idle: 'transparent',
}

const SCREEN_COLOR: Record<CharacterState, string> = {
  offline: '#111',
  error: '#2a0d0d',
  program_running: '#0d1530',
  just_traded: '#0f1f0a',
  ai_thinking: '#0d1525',
  holding_profit: '#0a1f0e',
  holding_loss: '#1f0a0a',
  idle: '#0d1520',
}

export default function Workstation({
  traderName,
  equity,
  unrealizedPnl,
  positionCount,
  avatarPresetId,
  state,
}: WorkstationProps) {
  const glowColor = STATE_GLOW[state]
  const screenColor = SCREEN_COLOR[state]
  const isOff = state === 'offline'

  return (
    <div className="relative flex flex-col items-center group" style={{ width: 140 }}>
      {/* HUD above — expands on hover */}
      <TraderHUD
        name={traderName}
        equity={equity}
        unrealizedPnl={unrealizedPnl}
        positionCount={positionCount}
        state={state}
      />

      {/* Cubicle container */}
      <div
        className="relative rounded-md overflow-hidden"
        style={{
          width: 130,
          height: 130,
          background: '#15171c',
          boxShadow: glowColor !== 'transparent'
            ? `0 0 20px 4px ${glowColor}`
            : 'none',
        }}
      >
        {/* Cubicle back wall */}
        <div
          className="absolute top-0 left-0 right-0"
          style={{
            height: 6,
            background: 'linear-gradient(180deg, #2a2d35, #1e2028)',
            borderBottom: '1px solid #333640',
          }}
        />
        {/* Cubicle left wall */}
        <div
          className="absolute top-0 left-0 bottom-0"
          style={{
            width: 4,
            background: 'linear-gradient(90deg, #2a2d35, #1e2028)',
            borderRight: '1px solid #333640',
          }}
        />
        {/* Cubicle right wall */}
        <div
          className="absolute top-0 right-0 bottom-0"
          style={{
            width: 4,
            background: 'linear-gradient(270deg, #2a2d35, #1e2028)',
            borderLeft: '1px solid #333640',
          }}
        />

        {/* Monitor */}
        <div className="absolute" style={{ top: 10, left: '50%', transform: 'translateX(-50%)' }}>
          {/* Monitor frame */}
          <div
            style={{
              width: 52,
              height: 36,
              background: '#1a1a2e',
              borderRadius: 3,
              padding: 3,
              position: 'relative',
            }}
          >
            {/* Screen */}
            <div
              style={{
                width: '100%',
                height: '100%',
                background: screenColor,
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {!isOff && unrealizedPnl !== null && unrealizedPnl !== 0 && (
                <span
                  className="font-mono font-bold"
                  style={{
                    fontSize: 8,
                    color: unrealizedPnl > 0 ? '#22c55e' : '#ef4444',
                    lineHeight: 1,
                  }}
                >
                  {unrealizedPnl > 0 ? '+' : ''}{unrealizedPnl.toFixed(1)}
                </span>
              )}
              {!isOff && (unrealizedPnl === null || unrealizedPnl === 0) && (
                <div style={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
                  {[3, 5, 4, 6, 3, 5, 7, 4].map((h, i) => (
                    <div
                      key={i}
                      style={{
                        width: 2,
                        height: h,
                        background: '#1e4d3a',
                        borderRadius: 1,
                      }}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
          {/* Monitor stand */}
          <div style={{ width: 8, height: 6, background: '#1a1a2e', margin: '0 auto' }} />
          <div style={{ width: 20, height: 3, background: '#1a1a2e', margin: '0 auto', borderRadius: 1 }} />
        </div>

        {/* Desk surface */}
        <div
          className="absolute"
          style={{
            top: 56,
            left: 10,
            right: 10,
            height: 10,
            background: 'linear-gradient(180deg, #3d2b1f, #2a1e15)',
            borderRadius: 2,
          }}
        />

        {/* Character — sits behind desk */}
        <div
          className="absolute"
          style={{
            bottom: 2,
            left: '50%',
            transform: 'translateX(-50%)',
          }}
        >
          <PixelCharacter
            presetId={avatarPresetId}
            state={state}
            scale={2}
          />
        </div>
      </div>
    </div>
  )
}
