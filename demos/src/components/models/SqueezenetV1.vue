<template>
  <div class="demo squeezenet-v1">
    <div class="title">
      <span>SqueezeNet v1.1, trained on ImageNet</span>
    </div>
    <mdl-spinner v-if="modelLoading && loadingProgress < 100"></mdl-spinner>
    <div class="loading-progress" v-if="modelLoading && loadingProgress < 100">
      Loading...{{ loadingProgress }}%
    </div>
    <div class="top-container" v-if="!modelLoading">
      <div class="input-container">
        <div class="input-label">Enter a valid image URL or select an image from the dropdown:</div>
        <div class="image-url">
          <mdl-textfield
            floating-label="enter image url"
            v-model="imageURLInput"
            spellcheck="false"
            @keyup.native.enter="onImageURLInputEnter"
          ></mdl-textfield>
          <span>or</span>
          <mdl-select
            label="select image"
            id="image-url-select"
            v-model="imageURLSelect"
            :options="imageURLSelectList"
            style="width:200px;"
          ></mdl-select>
        </div>
      </div>
      <div class="controls">
        <mdl-switch v-model="useGpu" :disabled="modelLoading || modelRunning || !hasWebgl">Use GPU</mdl-switch>
        <mdl-switch v-model="showComputationFlow" :disabled="modelLoading || modelRunning">Show computation flow</mdl-switch>
      </div>
    </div>
    <div class="columns input-output" v-if="!modelLoading">
      <div class="column input-column">
        <div class="loading-indicator">
          <mdl-spinner v-if="imageLoading || modelRunning"></mdl-spinner>
          <div class="error" v-if="imageLoadingError">Error loading URL</div>
        </div>
        <div class="canvas-container">
          <canvas id="input-canvas" width="227" height="227"></canvas>
        </div>
      </div>
      <div class="column output-column">
        <div class="output">
          <div class="output-class"
            :class="{ predicted: i === 0 && outputClasses[i].probability.toFixed(2) > 0 }"
            v-for="i in [0, 1, 2, 3, 4]"
          >
            <div class="output-label">{{ outputClasses[i].name }}</div>
            <div class="output-bar"
              :style="{width: `${Math.round(100 * outputClasses[i].probability)}px`, background: `rgba(27, 188, 155, ${outputClasses[i].probability.toFixed(2)})` }"
            ></div>
            <div class="output-value">{{ Math.round(100 * outputClasses[i].probability) }}%</div>
          </div>
        </div>
      </div>
    </div>
    <div class="layer-results-container"  v-if="!modelLoading" id="results-container">
    	<div id="webgl_container"></div>
    </div>
  </div>
</template>

<script>
import loadImage from 'blueimp-load-image'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import filter from 'lodash/filter'
import * as utils from '../../utils'
require('../../js/controls/myOrbitControls.js')
import { IMAGE_URLS } from '../../data/sample-image-urls'
import { ARCHITECTURE_DIAGRAM, ARCHITECTURE_CONNECTIONS } from '../../data/squeezenet-v1.1-arch'

const MODEL_FILEPATHS_DEV = {
  model: '/demos/data/squeezenet_v1.1/squeezenet_v1.1.json',
  weights: '/demos/data/squeezenet_v1.1/squeezenet_v1.1_weights.buf',
  metadata: '/demos/data/squeezenet_v1.1/squeezenet_v1.1_metadata.json'
}
const MODEL_FILEPATHS_PROD = {
  model: 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1.json',
  weights: 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1_weights.buf',
  metadata: 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1_metadata.json'
}
const MODEL_CONFIG = { filepaths: process.env.NODE_ENV === 'production' ? MODEL_FILEPATHS_PROD : MODEL_FILEPATHS_DEV }

export default {
  props: ['hasWebgl'],

  data: function() {
    return {
      useGpu: this.hasWebgl,
      showComputationFlow: true,
      model: new KerasJS.Model(
        Object.assign({ gpu: this.hasWebgl, pipeline: false, layerCallPauses: true }, MODEL_CONFIG)
      ), // eslint-disable-line
      modelLoading: true,
      modelRunning: false,
      imageURLInput: '',
      imageURLSelect: null,
      imageURLSelectList: IMAGE_URLS,
      imageLoading: false,
      imageLoadingError: false,
      output: null,
      architectureDiagram: ARCHITECTURE_DIAGRAM,
      architectureConnections: ARCHITECTURE_CONNECTIONS,
      architectureDiagramPaths: []
    }
  },

  watch: {
    imageURLSelect: function(value) {
      this.imageURLInput = value
      this.loadImageToCanvas(value)
    },
    useGpu: function(value) {
      this.model.toggleGpu(value)
    },
    showComputationFlow: function(value) {
      this.model.layerCallPauses = value
    }
  },

  computed: {
    loadingProgress: function() {
      return this.model.getLoadingProgress()
    },
    architectureDiagramRows: function() {
      const rows = []
      for (let row = 0; row < 50; row++) {
        rows.push(filter(this.architectureDiagram, { row }))
      }
      return rows
    },
    layersWithResults: function() {
      // store as computed property for reactivity
      return this.model.layersWithResults
    },
    outputClasses: function() {
      if (!this.output) {
        const empty = []
        for (let i = 0; i < 5; i++) {
          empty.push({ name: '-', probability: 0 })
        }
        return empty
      }
      return utils.imagenetClassesTopK(this.output, 5)
    }
  },

  mounted: function() {
    this.model.ready().then(() => {
      this.modelLoading = false
      this.$nextTick(() => {
        this.initWebgl();
        //this.getIntermediateResults();
      })
    })
  },

  methods: {

	initWebgl: function() {			
		const container = document.getElementById("webgl_container");
        this.container = container;

        this.originalWidth = 0.95*container.offsetWidth ;
		this.originalHeight = 1000;
        this.rotatingCam = false;
		this.posX = [];
		this.posY = [];
		this.posZ = [];
		this.layerNum = [];
		this.points = null;
		this.pickedPoint = null;
		this.lines = null;
		this.clicked = false;

		const scene = new THREE.Scene();
        //scene.background = new THREE.Color( 0xff0000 );
        this.scene = scene;

		const highlightBox = new THREE.Mesh(
			new THREE.BoxGeometry( 12,12,12 ),
			new THREE.MeshLambertMaterial( { color: 0xffff00 }
		) );
		highlightBox.visible = false;
		scene.add( highlightBox );
        this.highlightBox = highlightBox;

		var light3 = new THREE.AmbientLight( 0xffffff);
		light3.name = "light";
		scene.add( light3 );

	    const camera = new THREE.PerspectiveCamera( 75, this.originalWidth / this.originalHeight, 1, 50000 );
	    camera.position.z = 5000;
        this.camera = camera;

	    const renderer = new THREE.WebGLRenderer({ antialias: true});
        renderer.setClearColor( 0xFFAAFF );
		renderer.setPixelRatio( window.devicePixelRatio );
		renderer.sortObjects = false;
	    renderer.setSize( this.originalWidth, this.originalHeight );
        this.renderer = renderer;

		var cameraControls = new THREE.OrbitControls( this.camera, renderer.domElement );
		cameraControls.target.set( 0, 0, 0 );

		var loader = new THREE.FontLoader();
		loader.load( 'https://raw.githubusercontent.com/rollup/three-jsnext/master/examples/fonts/droid/droid_sans_bold.typeface.json',this.setFont);

	    container.appendChild( renderer.domElement );
		renderer.domElement.addEventListener( 'mousemove', this.onMouseMove );
		renderer.domElement.addEventListener( 'mousedown', this.onMouseDown );
		renderer.domElement.addEventListener( 'click', this.onClick );
		renderer.domElement.addEventListener( 'mouseup', this.onMouseUp );
		window.addEventListener( 'resize', this.onWindowResize, false );
		this.animate();
	},

	animate: function() {
		requestAnimationFrame( this.animate );
		this.render();
    },
 
	render: function() {
		this.renderer.render( this.scene, this.camera );
	},
	setFont: function( font ) {
		this.font = font;
	},
		
	onWindowResize: function( e ) {
			//render();
	},
		
	onMouseDown: function( e ) {
		this.rotatingCam = true;
		this.clicked = true;
	},
	
	onClick: function( e ) {
		//console.log('click');
	},
	
	onMouseUp: function( e ) {
		//console.log('mouse up');
		this.rotatingCam = false;	
		this.clicked = false;
		
	},

	onMouseMove: function( e ) {
		event.preventDefault();
	},


    onImageURLInputEnter: function(e) {
      this.imageURLSelect = null
      this.loadImageToCanvas(e.target.value)
    },
    loadImageToCanvas: function(url) {
      if (!url) {
        this.clearAll()
        return
      }

      this.imageLoading = true
      loadImage(
        url,
        img => {
          if (img.type === 'error') {
            this.imageLoadingError = true
            this.imageLoading = false
          } else {
            // load image data onto input canvas
            const ctx = document.getElementById('input-canvas').getContext('2d')
            ctx.drawImage(img, 0, 0)
            this.imageLoadingError = false
            this.imageLoading = false
            this.modelRunning = true
			
			  while(this.scene.children.length > 0) { 
				this.scene.remove(this.scene.children[0]); 
			  }
            // model predict
            this.$nextTick(function() {
              setTimeout(() => {
                this.runModel()
              }, 200)
            })
          }
        },
        { maxWidth: 227, maxHeight: 227, cover: true, crop: true, canvas: true, crossOrigin: 'Anonymous' }
      )
    },
    runModel: function() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData

      // data processing
      // see https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.subseq(dataTensor.pick(null, null, 2), 103.939)
      ops.subseq(dataTensor.pick(null, null, 1), 116.779)
      ops.subseq(dataTensor.pick(null, null, 0), 123.68)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 2))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 0))

      const inputData = { input_1: dataProcessedTensor.data }
      this.model.predict(inputData).then(outputData => {
        this.output = outputData['loss'];
        this.modelRunning = false;
		this.getIntermediateResults();
      })
    },

    getIntermediateResults: function() {
      let results = []
      for (let [name, layer] of this.model.modelLayersMap.entries()) {
        if (name.startsWith('dropout')) continue

        const layerClass = layer.layerClass || ''

        let images = []
		if (layer.result) {
			if(name=='input_1') {
		      images = [utils.image3Dtensor(layer.result.tensor)]
			} else if (layer.result.tensor.shape.length === 3) {
		      images = utils.unroll3Dtensor(layer.result.tensor)
		    } else if (layer.result.tensor.shape.length === 2) {
		      images = [utils.image2Dtensor(layer.result.tensor)]
		    } else if (layer.result.tensor.shape.length === 1) {
		      images = [utils.image1Dtensor(layer.result.tensor)]
		    }
		    results.push({ name, layerClass, images })
		}
      }
      this.layerResultImages = results
      setTimeout(() => {
        this.displayOutput()
      }, 0)
    },
    clearAll: function() {
      this.modelRunning = false
      this.imageURLInput = null
      this.imageURLSelect = null
      this.imageLoading = false
      this.imageLoadingError = false
      this.output = null

      this.model.layersWithResults = []

      const ctx = document.getElementById('input-canvas').getContext('2d')
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    },
	displayOutput: function() {
        if(this.layerResultImages.length <= 1) {
			return;	
		}
		var geometry = new THREE.Geometry();
		var matrix = new THREE.Matrix4();
 		var quaternion = new THREE.Quaternion();
		var materials = [];
	    var layerX = 0; var layerY = 0; var layerZ = 0; var height = 0; var width = 0; var fc = false; var spacing = 0; var len = 0; var xPos = 0; var yPos = 0; 
		this.layerResultImages.forEach((result, layerNum) => {
			var layer =  this.architectureDiagram[layerNum];
	        var images = result.images;
	        len = Math.ceil(Math.sqrt(images.length));
	        layerZ = -layer.row * 450;
			layerY = 0;
			if(layerNum < 650 && layerNum > 1) {
				if(layer.row === this.architectureDiagram[layerNum+1].row){
	    			layerY = - len/2 *images[0].width*1.5;
				}
				else if(this.architectureDiagram[layerNum-1].row === layer.row){
	    			layerY = len/2 *images[0].width*1.5;
				}
			}
	        if(images.length==1 && images[0].height==1) {
	            fc = true;
	            spacing = 18;
			} else {
	            fc = false;
	            spacing = 10;
	    		xPos = layerY - len/2 *images[0].width;
				yPos = layerX + len/2 *images[0].height;
				var text = this.createLabel(result.name, -350, yPos+700, layerZ, 30, "white");
				this.scene.add(text);
			}
	    	images.forEach((image, imageNum) => {
				var geom = new THREE.BoxGeometry( image.width,image.height,image.width );
	            width = image.width;
	            height = image.height;
	            if(fc) {  
	                len = Math.ceil(Math.sqrt(width));
	            	width = len;
	            	height = Math.ceil(width/height);
	        		xPos = layerX - len/2 * 18;
	    			yPos = layerY + len/2 * 18;
					//var text = this.createLabel(result.name, -350, yPos+700, layerZ, 30, "white");
					//this.scene.add(text);
	            } else {
	        		xPos = layerY + (-len/2 + imageNum%len)*image.width*1.1;
	    			yPos = layerX - (-len/2 + Math.floor(imageNum/len))*image.height*1.1;
	            }
				var texture = new THREE.Texture(image);
				texture.needsUpdate = true;
				// material
				var material = new THREE.MeshPhongMaterial( { map: texture } );
				materials.push(material);
				var position = new THREE.Vector3();
				position.x = xPos;
				position.y = yPos;
				position.z = layerZ;

				var rotation = new THREE.Euler();
				rotation.x = 0;
				rotation.y = 0;
				rotation.z = 0;

				var scale = new THREE.Vector3();
				scale.x = 1;
				scale.y = 1;
				scale.z = 1;
				
				quaternion.setFromEuler( rotation, false );
				matrix.compose( position, quaternion, scale );

				//geometry.merge( geom, matrix, materials.length-1);
			    var smaterial = new THREE.SpriteMaterial( { map: texture} );
				var sprite = new THREE.Sprite( smaterial );
				sprite.position.set( xPos,yPos,layerZ );
				sprite.scale.set( image.width,image.height, 1.0 ); // imageWidth, imageHeight
				this.scene.add( sprite );
			});
		});
		//var drawnObject = new THREE.Mesh( geometry, materials);//new THREE.MeshFaceMaterial(materials) );
        //drawnObject.geometry.computeFaceNormals();
        //drawnObject.geometry.computeVertexNormals();
		//drawnObject.name = 'cubes';
		//this.scene.add( drawnObject );
	},
    createLabel: function(text, x, y, z, size, color) {
		var materialFront = new THREE.MeshBasicMaterial( { color: color} );
		var materialSide = new THREE.MeshBasicMaterial( { color: 0x000088 } );
		var materialArray = [ materialFront, materialSide ];
		var textGeom = new THREE.TextGeometry( text, 
		{
			size: size, height: 4, curveSegments: 3, font: this.font,
			bevelThickness: 1, bevelSize: 2, bevelEnabled: true,
			material: 0, extrudeMaterial: 1
		});
	
		var textMesh = new THREE.Mesh(textGeom, materialArray );
		textGeom.computeBoundingBox();
    	textGeom.textWidth = textGeom.boundingBox.max.x - textGeom.boundingBox.min.x;
		textMesh.position.set( x-textGeom.textWidth/2, y, z );
		return textMesh;
	}
  }
}
</script>

<style scoped>
@import '../../variables.css';

.demo.squeezenet-v1 {
  & .top-container {
    margin: 10px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;

    & .input-container {
      & .input-label {
        font-family: var(--font-cursive);
        font-size: 16px;
        color: var(--color-lightgray);
        text-align: left;
        user-select: none;
        cursor: default;
      }

      & .image-url {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: flex-start;
        position: relative;

        & span {
          margin: 0 10px;
          font-family: var(--font-cursive);
          font-size: 16px;
          color: var(--color-lightgray);
        }
      }
    }

    & .controls {
      width: 250px;
      margin-left: 40px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;

      & .mdl-switch {
        margin-bottom: 5px;
      }
    }
  }

  & .columns.input-output {
    max-width: 800px;
    margin: 0 auto;

    & .column {
      display: flex;
      align-items: center;
      justify-content: center;
    }

    & .column.input-column {
      position: relative;

      & .loading-indicator {
        position: absolute;
        top: 0;
        left: -10px;
        display: flex;
        flex-direction: column;
        align-self: flex-start;

        & .mdl-spinner {
          margin: 20px;
          align-self: center;
        }

        & .error {
          color: var(--color-error);
          font-size: 14px;
          font-family: var(--font-sans-serif);
          margin: 20px;
        }
      }

      & .canvas-container {
        display: inline-flex;
        justify-content: flex-end;

        & canvas {
          background: whitesmoke;
        }
      }
    }

    & .column.output-column {
      & .output {
        width: 370px;
        height: 160px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;

        & .output-class {
          display: flex;
          flex-direction: row;
          align-items: center;
          justify-content: center;
          padding: 6px 0;

          & .output-label {
            text-align: right;
            width: 200px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-family: var(--font-monospace);
            font-size: 18px;
            color: var(--color-lightgray);
            padding: 0 6px;
            border-right: 2px solid var(--color-green-lighter);
          }

          & .output-bar {
            height: 8px;
            transition: width 0.2s ease-out;
          }

          & .output-value {
            text-align: left;
            margin-left: 5px;
            font-family: var(--font-monospace);
            font-size: 14px;
            color: var(--color-lightgray);
          }
        }

        & .output-class.predicted {
          & .output-label {
            color: var(--color-green);
            border-left-color: var(--color-green);
          }

          & .output-value {
            color: var(--color-green);
          }
        }
      }
    }
  }

  & .architecture-container {
    min-width: 700px;
    max-width: 900px;
    margin: 0 auto;
    position: relative;

    & .layers-row {
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;
      margin-bottom: 5px;
      position: relative;
      z-index: 1;

      & .layer-column {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 5px;

        & .layer {
          display: inline-block;
          background: whitesmoke;
          border: 2px solid whitesmoke;
          border-radius: 5px;
          padding: 2px 10px 0px;

          & .layer-class-name {
            color: var(--color-green);
            font-size: 14px;
            font-weight: bold;
          }

          & .layer-details {
            color: #999999;
            font-size: 12px;
            font-weight: bold;
          }
        }

        & .layer.has-result {
          border-color: var(--color-green);
        }
      }
    }

    & .architecture-connections {
      position: absolute;
      top: 0;
      left: 0;
      z-index: 0;

      & path {
        stroke-width: 4px;
        stroke: #AAAAAA;
        fill: none;
      }
    }
  }
}
</style>
