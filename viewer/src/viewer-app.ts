import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {Coding} from "./types";

// const logo = new URL('../../assets/open-wc-logo.svg', import.meta.url).href;

enum ViewType {
  Manual,
  Automatic,
  Overlap,
  Difference,
}

@customElement('viewer-app')
export class ViewerApp extends LitElement {
  @property({ type: String }) title = 'My app';

  @property({ type: Object }) data?: Coding[];

  @property({ type: ViewType }) overlap?: ViewType = ViewType.Manual;

  // language=css
  static styles = css`
    :host {
      min-height: 100vh;
      display: inline;
      flex-direction: column;
      justify-content: flex-start;
      color: #1a2b42;
      max-width: 1200px;
      margin: 0 auto;
      text-align: left;
    }
    
    div[name='Experienced QoC'] {
      background-color: #FFCDD2;
    }
    
    div[name='Context'] {
      background-color: #E1BEE7;
    }
    
    div[name='Experiences'] {
      background-color: #BBDEFB;
    }
    
    div[name='Expectations'] {
      background-color: #DCEDC8;
    }
        
    div[label='Positive'] {
      background-color: #88FF88;
    }
    
    div[label='Negative'] {
      background-color: #FFAAAA;
    }
    
    .label-el {
      display: inline-block;
      padding: 2px;
      border-radius: 5px;
    }
    
    .label-container {
      display: inline-flex;
      justify-content: center;
      align-content: center;
      vertical-align: bottom;
      flex-direction: column;
      padding: 4px;
      margin: 2px 0;
      border-radius: 5px;
    }
    
    .label {
      display: block;
      padding: 3px 0;
      font-size: 8px;
    }
  `;

  async connectedCallback() {
    super.connectedCallback();

    this.data = await fetch('/assets/index_resident.json').then(r => r.json());
  }

  render() {
    return html`
      <div style="max-width: 960px; margin: auto;">
        <p></p>
        <fieldset>
          <legend>Selecteer een transcript:</legend>
          <div>
            <input type="radio" id="huey" name="drone" value="index_resident" checked @change=${async (e: CustomEvent) => {
              this.data = await fetch('/assets/index_resident.json').then(r => r.json());
            }}>
            <label for="index_resident">Resident</label>
          </div>
          
          <div>
            <input type="radio" id="louie" name="drone" value="index_care_professional" @change=${async (e: CustomEvent) => {
              this.data = await fetch('/assets/index_care_professional.json').then(r => r.json());
            }}>
            <label for="index_care_professional">Care Professional</label>
          </div>
      
          <div>
            <input type="radio" id="dewey" name="drone" value="index_family" @change=${async (e: CustomEvent) => {
              this.data = await fetch('/assets/index_family.json').then(r => r.json());
            }}>
            <label for="index_family">Family</label>
          </div>
      </fieldset>
      <fieldset>
          <legend>Mode:</legend>
          <div>
            <input type="radio" id="manual" name="overlap" value="manual" checked @change=${async (e: CustomEvent) => {
              this.overlap = ViewType.Manual;
            }}>
            <label for="manual">Handmatig</label>
          </div>
        
          <div>
            <input type="radio" id="automatic" name="overlap" value="automatic" @change=${async (e: CustomEvent) => {
              this.overlap = ViewType.Automatic;
            }}>
            <label for="automatic">Text Mining</label>
          </div>
        
          <div>
            <input type="radio" id="overlap" name="overlap" value="overlap" @change=${async (e: CustomEvent) => {
              this.overlap = ViewType.Overlap;
            }}>
            <label for="overlap">Overlap</label>
          </div>
          
          <div>
            <input type="radio" id="difference" name="overlap" value="difference" @change=${async (e: CustomEvent) => {
              this.overlap = ViewType.Difference;
            }}>
            <label for="difference">Verschil</label>
          </div>
      </fieldset>
      <br/>
        
      ${this.renderLines()}
      <p></p>
      <p></p>
      </div>
    `;
  }

  getPercentPositive() : number {
    let pos = 0;
    let neg = 0;

    this.data?.forEach(el => {
      if (el.label.includes("Positive"))
        pos += 1;

      if (el.label.includes("Negative"))
        neg += 1;
    });

    return pos / (pos + neg) * 100;
  }

  private static getBackground(labels: boolean[]) : string {
    let out = '#';

    for (let i = 0; i < 3; i++)
      out += labels[i] ? 'FF' : 'AA';

    if (out === '#000000')
      return 'transparent';

    return out;
  }

  private static getBorder(labels: boolean[]) : string {
    let out = '1px solid #';

    if (this.getBackground(labels) == "#FFFFFF") {
      console.log("#FFFFFF");
      // out += "000";
    } else if (labels[3]) {
      out += "000";
    }

    return out;
  }

  private static average(array: number[]) : number {
    const sum = array.reduce((a, b) => a + b, 0);
    return (sum / array.length) || 0;
  }

  private renderLines() {
    let index = 0;

    return this.data?.map(item => {
      const newLine = item.text[1] === ":" || item.text.length > 132 || index === 0;

      index += 1;

      const labels = item.label.map(l => l.split(" (")[0]);
      const overlap = this.findDuplicateAndUniqueValues(labels);
      let labelValues = overlap.duplicates;
      if (this.overlap === ViewType.Difference) {
        labelValues = this.findUniqueValues(labels).map(el => item.label[el.index]);
      } else if (this.overlap === ViewType.Manual) {
        labelValues = item.label.filter(l => l.includes("(M)")).map(l => l.split(" (")[0]);
      } else if (this.overlap === ViewType.Automatic) {
        labelValues = item.label.filter(l => l.includes("(A)")).map(l => l.split(" (")[0]);
      }

      return html`
        
        ${newLine ? html`<br/>` : ''}
        <div class="label-container" style="background: ${labelValues.length > 0 ? '#00000011' : ''}">
          <div class="label">
            <div class="label-el" label="0">
              ${index}
            </div>
            ${labelValues.map(l => html`
              <div class="label-el" label="${l}" name="${l.split(" (")[0]}">
                  ${l}
              </div>
            `)}
          </div>
          <div class="text">${item.text}</div>
        </div>
      `
    });
  }

  findDuplicateAndUniqueValues(strings: string[]): { duplicates: string[], unique: string[] } {
    // Create a frequency map to keep track of the frequency of each string
    const frequencyMap: { [key: string]: number } = {};

    // Iterate through the array of strings and add each string to the frequency map
    for (const str of strings) {
      if (frequencyMap[str]) {
        frequencyMap[str] += 1;
      } else {
        frequencyMap[str] = 1;
      }
    }

    // Create an array to store the duplicate values
    const duplicates: string[] = [];
    // Create an array to store the unique values
    const unique: string[] = [];

    // Iterate through the frequency map and add any strings with a frequency of 1 to the unique array, and any strings with a frequency greater than 1 to the duplicates array
    for (const str in frequencyMap) {
      if (frequencyMap[str] > 1) {
        duplicates.push(str);
      } else {
        unique.push(str);
      }
    }

    return { duplicates, unique };
  }

  findUniqueValues(strings: string[]): { value: string, index: number }[] {
    // Create a frequency map to keep track of the frequency of each string
    const frequencyMap: { [key: string]: number } = {};

    // Iterate through the array of strings and add each string to the frequency map
    for (const str of strings) {
      if (frequencyMap[str]) {
        frequencyMap[str] += 1;
      } else {
        frequencyMap[str] = 1;
      }
    }

    // Create an array to store the indices of the unique values
    const uniqueValues: { value: string, index: number }[] = [];

    // Iterate through the input array and add the index and value of any unique values to the uniqueValues array
    for (let i = 0; i < strings.length; i++) {
      if (frequencyMap[strings[i]] === 1) {
        uniqueValues.push({ value: strings[i], index: i });
      }
    }

    return uniqueValues;
  }
}