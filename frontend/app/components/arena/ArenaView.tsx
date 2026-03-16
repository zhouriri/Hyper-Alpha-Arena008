import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import TradingFloor, { type TraderData } from './TradingFloor'
import type { CharacterState } from './pixelData/characters'
import type { Position } from '@/components/portfolio/HyperliquidMultiAccountSummary'
import type { HyperliquidEnvironment } from '@/lib/types/hyperliquid'

interface AccountData {
  account_id: number
  account_name: string
  exchange?: string
  avatar_preset_id?: number | null
  auto_trading_enabled?: boolean
}

interface AccountBalance {
  accountId: number
  accountName: string
  exchange: string
  balance: {
    totalEquity: number
    marginUsagePercent: number
  } | null
  error: string | null
}

interface ArenaViewProps {
  accounts: AccountData[]
  positions: Position[]
  accountBalances: AccountBalance[]
  environment: HyperliquidEnvironment
}

function deriveState(
  account: AccountData,
  accountPositions: Position[],
  balance: AccountBalance | undefined,
): CharacterState {
  if (account.auto_trading_enabled === false) return 'offline'
  if (balance?.error) return 'error'

  const totalPnl = accountPositions.reduce((sum, p) => sum + p.unrealized_pnl, 0)
  if (accountPositions.length > 0) {
    return totalPnl >= 0 ? 'holding_profit' : 'holding_loss'
  }

  return 'idle'
}

export default function ArenaView({
  accounts,
  positions,
  accountBalances,
}: ArenaViewProps) {
  const { t } = useTranslation()

  const traders: TraderData[] = useMemo(() => {
    return accounts.map((acc) => {
      const accPositions = positions.filter(
        (p) => p.account_id === acc.account_id
      )
      const balance = accountBalances.find(
        (b) => b.accountId === acc.account_id
      )
      const totalPnl = accPositions.reduce((sum, p) => sum + p.unrealized_pnl, 0)
      const state = deriveState(acc, accPositions, balance)

      return {
        accountId: acc.account_id,
        accountName: acc.account_name,
        avatarPresetId: acc.avatar_preset_id ?? null,
        equity: balance?.balance?.totalEquity ?? null,
        unrealizedPnl: totalPnl || null,
        positionCount: accPositions.length,
        state,
      }
    })
  }, [accounts, positions, accountBalances])

  if (accounts.length === 0) {
    return (
      <div className="flex items-center justify-center h-full min-h-[280px] rounded-lg bg-card border border-border">
        <div className="text-sm text-muted-foreground">
          {t('dashboard.noAccountConfigured', 'No account configured')}
        </div>
      </div>
    )
  }

  return <TradingFloor traders={traders} />
}
