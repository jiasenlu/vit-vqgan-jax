from models.modeling import *
from models.checkpoint import f32_to_bf16, bf16_to_f32
from training.train_state import TrainState_v2

class Generator(Model):
    """
    Generator
    """
    def __call__(self, x, *, train):
        """
        Does a forward pass for pretraining
        :param batch: Everything from pretraining
        :return:
        """
        rec, diff = self.model(x, train=train)
        return rec, diff

class Discriminator(Model):
    """
    Discriminator
    """
    def __call__(self, x, *, train):
        """
        Does a forward pass for pretraining
        :param batch: Everything from pretraining
        :return:
        """
        output = self.model(x, train=train)
        return output

class LPIPS(Model):
    """
    LPIPS
    """
    def __call__(self, x0, x1, *, train):
        """
        Does a forward pass for pretraining
        :param batch: Everything from pretraining
        :return:
        """
        output = self.model(x0, x1, train=train)
        return output

def vanilla_d_loss(logits_fake, logits_real = None):
    loss_fake = jnp.mean(jax.nn.softplus(-logits_fake)) * 2 if logits_real is None else jnp.mean(jax.nn.softplus(logits_fake))
    loss_real = 0 if logits_real is None else jnp.mean(jax.nn.softplus(-logits_real))
    return 0.5 * (loss_real + loss_fake)

def hinge_d_loss(logits_fake, logits_real = None):
    loss_fake = jnp.mean(jax.nn.relu(1.0 + logits_fake))
    loss_real = jnp.mean(jax.nn.relu(1.0 - logits_real))
    return 0.5 * (loss_real + loss_fake)

def calculate_adaptive_weight():
    pass

def VQLPIPS(
    input,
    rec, 
    codebook_loss, 
    p_logits,
    g_logits,
    codebook_weight: float = 1.0,
    loggaussian_weight: float = 1.0,
    loglaplace_weight: float = 0.0,
    perceptual_weight: float = 0.1,
    adversarial_weight: float = 0.1,
    disc_g_start: int = 0,
    step: int = 0,
    ):

    loglaplace_loss =  jnp.mean(jnp.abs(rec - input))
    loggaussian_loss = jnp.mean((rec - input)**2)
    perceptual_loss = jnp.mean(p_logits)

    d_weight_factor = jnp.where(step > disc_g_start, adversarial_weight,0.0)
    adversarial_loss = jnp.mean(vanilla_d_loss(g_logits))

    nll_loss = loglaplace_weight * loglaplace_loss + loggaussian_weight * loggaussian_loss \
        + perceptual_weight * perceptual_loss + d_weight_factor * adversarial_loss

    loss = nll_loss + codebook_weight * codebook_loss
    return loss, (loglaplace_loss, loggaussian_loss, perceptual_loss, adversarial_loss, codebook_loss)

def train_step(state: TrainState_v2, batch, config=None):
    """
    Note: we'll compile this with pmap so no need to jit
    :param state:
    :param batch:
    :param use_bfloat16_grads: Whether to use bfloat16 for storing grads. I think it is probably OK considering
                               momentum is bfloat16 anyways. i'm just going to cast down (rounding down here rather
                               than to nearest or anything)
    :return:
    """
    def _loss_fn_nll(params):
        rec, codebook_loss = state.apply_fn_g(
            {'params': params},
            rngs=state.aux_rng_keys_g,
            x=batch['inputs'],
            train=True,
        )
        
        p_logits = state.apply_fn_p(
            {'params': state.params_p},
            rngs=state.aux_rng_keys_p,
            x0=batch['inputs'],
            x1=rec,
            train=False,
        )

        g_logits = state.apply_fn_d(
            {'params': state.params_d},
            rngs=state.aux_rng_keys_d,
            x=rec,
            train=False,            
        )      

        loss, aux_loss = VQLPIPS(
            batch['inputs'], 
            rec, 
            codebook_loss, 
            p_logits, 
            g_logits,
            codebook_weight = config['loss']['codebook_weight'],
            loggaussian_weight = config['loss']['loggaussian_weight'],
            loglaplace_weight = config['loss']['loglaplace_weight'],
            perceptual_weight = config['loss']['perceptual_weight'],
            adversarial_weight = config['loss']['adversarial_weight'],
            disc_g_start = config['loss']['disc_g_start'],
            step = state.step,
            )

        return jnp.mean(loss), (rec, aux_loss)

    def _loss_fn_d(params):
        rec, _ = state.apply_fn_g(
            {'params': state.params_g},
            rngs=state.aux_rng_keys_g,
            x=batch['inputs'],
            train=False,
        )
        rec = jax.lax.stop_gradient(rec)
        d_real_logit = state.apply_fn_d(
            {'params': params},
            rngs=state.aux_rng_keys_d,
            x=batch['inputs'],
            train=True,            
        )
        d_fake_logit = state.apply_fn_d(
            {'params': params},
            rngs=state.aux_rng_keys_d,
            x=rec,
            train=True,            
        )
        d_loss = vanilla_d_loss(d_fake_logit, d_real_logit)

        d_weight = jnp.where(state.step > config['loss']['disc_d_start'], 1.0, 0.0)
        d_loss = d_loss * d_weight

        return jnp.mean(d_loss), (jnp.mean(d_real_logit),jnp.mean(d_fake_logit))

    use_bfloat16_grads = config['model']['use_bfloat16']    
    params_g = state.params_g
    if use_bfloat16_grads:
        params_g = f32_to_bf16(params_g)

    grad_fn_nll = jax.value_and_grad(_loss_fn_nll, has_aux=True)
    output_nll, grads_nll = grad_fn_nll(params_g)
    loss_nll, aux_output = output_nll
    rec, aux_loss = aux_output
    loglaplace_loss, loggaussian_loss, p_loss, loss_g, c_loss = aux_loss

    # grad_fn_g = jax.value_and_grad(_loss_fn_g, has_aux=False)
    # loss_g, grads_g = grad_fn_g(params_g)
    
    # last_layer_nll = jnp.linalg.norm(grads_nll['model']['decoder_proj']['kernel'])
    # last_layer_g = jnp.linalg.norm(grads_g['model']['decoder_proj']['kernel'])

    # d_weight_factor = jnp.where(state.step > config['loss']['disc_g_start'], config['loss']['adversarial_weight'],0.0)
    # d_weight = last_layer_nll / (last_layer_g + 1e-4) * d_weight_factor

    # grads_g = jax.tree_map(lambda x: d_weight*x, grads_g)

    # grads = jax.tree_map(lambda x: jnp.nan_to_num(x, copy=False), grads)
    grads_nll = jax.lax.pmean(grads_nll, axis_name='batch')
    # grads_g = jax.lax.pmean(grads_g, axis_name='batch')

    # Cast up grads here (after the pmean) which reduces bandwidth maybe
    if use_bfloat16_grads:
        grads_nll = bf16_to_f32(grads_nll)
        # grads_g = bf16_to_f32(grads_g)

    state = state.apply_gradients_g(grads=grads_nll)
    # state = state.apply_gradients_g(grads=grads_g)

    loss_info = {}
    loss_info['loss'] = loss_nll
    loss_info['loglaplace_loss'] = loglaplace_loss
    loss_info['loggaussian_loss'] = loggaussian_loss
    loss_info['perceptual_loss'] = p_loss
    loss_info['codebook_loss'] = c_loss
    # loss_info['d_weight'] = d_weight

    if config['data']['task'] == 'audio':
        loss_info['image_rec'] = rec
        loss_info['image_ori'] = batch['inputs']
    else:
        loss_info['image_rec'] = (rec.clip(-1.0, 1.0) + 1.0) / 2.0
        loss_info['image_ori'] = (batch['inputs'] + 1.0) / 2.0

    # discriminator
    grad_fn_d = jax.value_and_grad(_loss_fn_d, has_aux=True)
    params_d = state.params_d
    if use_bfloat16_grads:
        params_d = f32_to_bf16(state.params_d)
    output_d, grads_d = grad_fn_d(params_d)

    loss_d, aux_output_d = output_d
    d_real_logit, d_fake_logit = aux_output_d

    loss_info['g_loss'] = loss_g
    loss_info['d_loss'] = loss_d
    loss_info['d_real_logit'] = d_real_logit
    loss_info['d_fake_logit'] = d_fake_logit

    if use_bfloat16_grads:
        grads_d = bf16_to_f32(grads_d)

    grads_d = jax.lax.pmean(grads_d, axis_name='batch')
    state = state.apply_gradients_d(grads=grads_d)

    # Average metrics over all replicas (maybe this isn't a great idea, idk)
    # loss_info = jax.lax.pmean(loss_info, axis_name='batch')
    loss_info = bf16_to_f32(loss_info)
    state = state.replace(step = state.step + 1)

    return state, loss_info