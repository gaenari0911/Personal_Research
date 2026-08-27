# R3 State Handling Contract

## Persistent variants B1/B2/B3

An episode execution is:

1. call `reset_episode_state(batch_size)` exactly once at episode start;
2. call `forward_step` for each observation, or one `forward_sequence` with
   `reset_episode=True`;
3. never reset at an official Step transition;
4. call `discard_episode_state()` at episode end.

`forward_step` fails if reset was omitted. A Step transition is expressed only
by B2/B3 language scheduling and has no access to the reset method. At B3 HOLD,
the language mask is false but `forward_step` still advances both causal-conv
and selective-SSM states.

## B0

B0 has no episode state. For every anchor `t`, it creates an unpadded local
window from `max(0,t-4)` through `t`, runs that window from zero Mamba state,
and keeps only the final normalized token. The next anchor starts from zero
again. Calling `reset_episode_state` or recurrent `forward_step` on B0 is an
error, preventing accidental conversion into a persistent model.

## Reset versus detach

R3 does not automatically detach or reset recurrent state at any chunk or Step
boundary. `MambaState.detach()` exists only as an explicit operation for a
future R4 decision. The only automatic detach is the frozen future CLIP target
inside InfoNCE; that is supervision, not temporal state.

Full-sequence and step-wise execution call the same `forward_step` recurrence.
The real 16-layer smoke produced a maximum absolute difference of
`9.5367431640625e-7`, within the `atol=1e-6, rtol=1e-5` test tolerance.

The returned state has two tensors per layer:

- causal convolution history `[batch, expand*d_model, d_conv-1]`;
- selective SSM state `[batch, expand*d_model, d_state]`.

Probe code must use returned final-normalized `temporal` (`z_t`, 128D), never
these raw caches.
