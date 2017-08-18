<template>
  <aside class="menu">
    <h1>Keras.js 3D Demos</h1>
    <p class="menu-label">Demos</p>
    <ul class="menu-list">
      <li :class="{ active: currentView === 'home' }">
        <a href="#/">Home</a>
      </li>
      <li :class="{ active: currentView === 'mnist-cnn' }">
        <a href="#/mnist-cnn">
          <span class="menu-item-heading">Basic Convnet</span>
          <span class="menu-item-subheading">MNIST</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'mnist-vae' }">
        <a href="#/mnist-vae">
          <span class="menu-item-heading">Convolutional VAE</span>
          <span class="menu-item-subheading">MNIST</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'mnist-acgan' }">
        <a href="#/mnist-acgan">
          <span class="menu-item-heading">AC-GAN</span>
          <span class="menu-item-subheading">MNIST</span>
        </a>
      </li>
     <!-- <li :class="{ active: currentView === 'resnet50' }">
        <a href="#/resnet50">
          <span class="menu-item-heading">Residual Network (50-layer)</span>
          <span class="menu-item-subheading">ImageNet</span>
        </a>
      </li>
      <li :class="{ active: currentView === 'inception-v3' }">
        <a href="#/inception-v3">
          <span class="menu-item-heading">Inception v3</span>
          <span class="menu-item-subheading">ImageNet</span>
        </a>
      </li>-->
      <li :class="{ active: currentView === 'squeezenet-v1.1' }">
        <a href="#/squeezenet-v1.1">
          <span class="menu-item-heading">SqueezeNet v1.1</span>
          <span class="menu-item-subheading">ImageNet</span>
        </a>
      </li>
     <!-- <li :class="{ active: currentView === 'imdb-bidirectional-lstm' }">
        <a href="#/imdb-bidirectional-lstm">
          <span class="menu-item-heading">Bidirectional LSTM</span>
          <span class="menu-item-subheading">IMDB</span>
        </a>
      </li>-->
    </ul>
  </aside>
</template>

<script>
export default {
  props: {
    currentView: { type: String, default: 'home' }
  }
}
</script>

<style scoped>
@import '../variables.css';

.menu {
  padding: 40px;
  margin: 20px;
  background: whitesmoke;
  border-radius: 10px;
  box-shadow: 5px 5px #CCCCCC;

  & h1 {
    color: #CCCCCC;
    font-family: var(--font-sans-serif);
    font-size: 1.5rem;
    user-select: none;
    cursor: default;
  }
}

.menu-label {
  user-select: none;
  cursor: default;
}

.menu-list {
  & li {
    color: var(--color-lightgray);
    border-left: 2px solid whitesmoke;
    margin-bottom: 5px;
  }

  & li.active {
    border-left: 2px solid var(--color-green);
  }

  & a {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;

    &:hover {
      color: var(--color-green);
      background-color: whitesmoke;

      & .github-logo {
        fill: var(--color-green);
      }
    }

    & span.menu-item-heading {
      margin-right: 10px;
    }

    & span.menu-item-subheading {
      color: #999999;
      font-size: 0.8em;
    }
  }
}

.menu-list.github, .menu-list.contact {
  & li {
    padding: 5px 10px;
  }

  & a {
    color: var(--color-green);
    padding: 0;
    display: inline-flex;
    background-color: none;
    transition: color 0.2s ease-in;

    & .github-logo {
      fill: var(--color-green);
      transition: fill 0.2s ease-in;
    }

    &:hover {
      color: var(--color-green-light);
      background: none;

      & .github-logo {
        fill: var(--color-green-light);
      }
    }
  }
}
</style>
