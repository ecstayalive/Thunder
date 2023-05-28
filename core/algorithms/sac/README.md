# SAC

# sac 算法可能的改进
- 最后更新动作分布$\pi$时，其是从reply buffer采样得到的，但是不得不承认，就是reply buffer是很大的，因此可能采样到的策略非常old，为了保证策略$\pi$的表现的上升趋势，在最后更新策略$\pi$时只使用最近的数据，可能算法表现更好。
- sac对reward scale非常敏感，因此可以想办法是其对reward scale参数不敏感。
