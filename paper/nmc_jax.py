import jax.numpy as jnp



def compute_single_pmi_jax(i, y_obs, outputs, xi, leave_out, log_likelihood_jax):
    ll = log_likelihood_jax(outputs, y_obs[i], None, xi)
    ll_gt = ll[i]
    n = ll.shape[0]

    if leave_out:
        ll_used = ll.at[i].set(-jnp.inf)
        count = n - 1

    else:
        ll_used = ll
        count = n

    max_ll = jnp.max(ll_used)
    log_evidence = max_ll + jnp.log(jnp.sum(jnp.exp(ll_used - max_ll))) - jnp.log(count)

    return ll_gt - log_evidence

