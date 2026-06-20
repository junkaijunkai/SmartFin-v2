export interface TransactionPayload {
  id: string
  date: string
  amount: number
  description: string
  merchant: string
  category?: string
  location?: string | null
}

export interface RunStreamRequest {
  message?: string
  monthly_income?: number
  current_date?: string
  use_sample_data?: boolean
  transactions?: TransactionPayload[]
  checkpoint_id?: string | null
  resume?: { confirmed: boolean; message?: string | null }
}

export interface PendingConfirmation {
  agent: string
  action: string
  summary: string
  details?: string[]
  categorisation_confidence?: string
  goal_extraction_confidence?: string
}

export type SSEEvent =
  | { node: '__pause__'; updates: { pending_confirmation: PendingConfirmation } }
  | { node: string; updates: Record<string, unknown> }
  | { node: string; reasoning_step: string }
  | { error: string }

export interface ThreadState {
  values: Record<string, unknown> | null
  next: string[]
  checkpoint_id: string | null
}

export interface TraceEntry {
  role: 'user' | 'agent'
  agent?: string
  content: string
  checkpointId?: string | null
  isReasoning?: boolean
  reasoning?: string
}

export interface Session {
  threadId: string
  title: string
  createdAt: number
  lastActivityAt: number
}
